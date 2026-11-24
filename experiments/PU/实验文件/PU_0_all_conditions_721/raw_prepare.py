"""Read-only raw archives; shared shift/downsampling/order tracking and noise projection."""
import torch
import argparse
from collections import defaultdict
import hashlib
import io
import os
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from common import (ROOT,PROJECT,CONDITIONS,RPM,BEARINGS,SOURCE,TARGETS,RAW_FS,DOWN_FS,
                    SPR,LENGTH,STRIDE,SAMPLE_SEED,NOISE_SEEDS,sha,write_json)


def read_mat(archive,member):
    tool = os.environ.get("PU_RAR_TOOL") or shutil.which("bsdtar") or "tar"
    payload=subprocess.run([tool,"-xOf",str(archive),member],capture_output=True,check=True).stdout
    mat=loadmat(io.BytesIO(payload),simplify_cells=True)
    record=next(value for key,value in mat.items() if not key.startswith("__"))
    y={item["Name"]:item for item in record["Y"]}
    if y["vibration_1"]["XIndex"]!=y["phase_current_1"]["XIndex"]:
        raise ValueError("Sensor clocks differ")
    times=np.asarray(record["X"][int(y["vibration_1"]["XIndex"])-1]["Data"],dtype="float64")
    values=np.stack([np.asarray(y[name]["Data"],dtype="float64") for name in ("vibration_1","phase_current_1")])
    if values.shape!=(2,len(times)) or not (np.diff(times)>0).all() or not np.isfinite(values).all():
        raise ValueError("Invalid raw samples or time axis")
    return values,times,hashlib.sha256(payload).hexdigest()


def down_and_order(values,fr,angle_count=None):
    down=resample_poly(values,1,4,axis=-1,window=("kaiser",5.),padtype="line")
    # Keep only output timestamps inside the actual trimmed 64kHz observation.
    count=min(down.shape[-1],int((values.shape[-1]-1)//4)+1)
    down=down[...,:count]
    if fr == 15:
        maximum = int(np.floor((count-1)/DOWN_FS*fr*SPR))+1
        if angle_count is None:
            angle_count = maximum
        if not 1 <= angle_count <= maximum:
            raise ValueError('Corrected order grid would extrapolate')
        corrected = resample_poly(down,3,5,axis=-1,window=('kaiser',5.),padtype='line')
        if corrected.shape[-1] < angle_count:
            raise ValueError('Insufficient resampled observation')
        return corrected[...,:angle_count]
    if fr != 25:
        raise ValueError('Only 900 and 1500 nominal rpm are supported')
    if angle_count is None:
        angle_count=int(np.floor((count-1)/DOWN_FS*fr*SPR))+1
    target=np.arange(angle_count)/(fr*SPR)
    clock=np.arange(count)/DOWN_FS
    if target[-1]>clock[-1] or target[0]<clock[0]:
        raise ValueError("Order resampling would extrapolate")
    if down.ndim==1:
        result=np.interp(target,clock,down)
    else:
        result=np.stack([np.interp(target,clock,row) for row in down])
    return result


def process(values,times,rpm):
    origin=max(0.,float(times[0]))
    count=int(np.floor((times[-1]-origin)*RAW_FS))+1
    clock=origin+np.arange(count)/RAW_FS
    if clock[-1]>times[-1]:
        raise ValueError("Shared time grid extrapolates")
    regular=np.stack([np.interp(clock,times,row) for row in values])
    fr=rpm/60
    tau=1/(2*4*fr)
    start=int(np.searchsorted(clock,origin+tau))
    delayed=clock[start:]-tau
    if delayed[0]<clock[0]:
        raise ValueError("Current shift requires unavailable history")
    paired=regular[:,start:].copy()
    paired[1]+=np.interp(delayed,clock,regular[1])
    result=down_and_order(paired,fr).astype("float32")
    raw_power=float(np.mean((regular[0]-regular[0].mean())**2))
    info=dict(raw_grid_samples=count,trim_start_sample=start,grid_origin_s=origin,
              rpm=rpm,electrical_hz=4*fr,shift_seconds=tau,angular_samples=result.shape[-1],
              raw_vibration_power=raw_power,fr=fr,
              effective_raw_fs=float((len(times)-1)/(times[-1]-times[0])))
    return result,info


def project_noise(info,seed,identity):
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(identity)]))
    noise=rng.standard_normal(info["raw_grid_samples"])
    noise-=noise.mean()
    noise/=np.sqrt(np.mean(noise**2))
    scaled=noise*np.sqrt(info["raw_vibration_power"])
    return down_and_order(scaled[info["trim_start_sample"]:],info["fr"],info["angular_samples"]).astype("float32")


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    state=dict(stage="preparing",started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
               condition_order=list(CONDITIONS),expected_windows=16000,source_windows=4000,
               target_windows_each=4000,formal_models=0,target_inference_accessed=False)
    write_json(out/"config.json",state)
    try:
        (out/"records").mkdir()
        x=np.lib.format.open_memmap(out/"signals.npy",mode="w+",dtype="float32",shape=(16000,2,LENGTH))
        templates=np.lib.format.open_memmap(out/"vibration_noise_templates.npy",mode="w+",dtype="float32",shape=(12000,5,LENGTH))
        records,windows=[],[]
        duplicate_groups=[]
        archive_hashes={}
        global_records={}
        for ci,condition in enumerate(CONDITIONS):
            for bi,code in enumerate(BEARINGS):
                archives=list((ROOT/"data/raw/PU").rglob(code+".rar"))
                if len(archives)!=1:
                    raise ValueError("Expected one archive per bearing")
                archive=archives[0]
                if code not in archive_hashes:
                    archive_hashes[code]=dict(path=str(archive),sha256=sha(archive))
                names=subprocess.run(["tar","-tf",str(archive)],capture_output=True,text=True,check=True).stdout.splitlines()
                names=sorted(name for name in names if Path(name).name.startswith(condition+"_"+code+"_") and name.endswith(".mat"))
                if len(names)!=20:
                    raise ValueError("Each bearing/condition must have twenty raw records")
                unique=[]
                seen={}
                for member in names:
                    values,times,raw_hash=read_mat(archive,member)
                    signal,info=process(values,times,RPM[condition])
                    full_hash=hashlib.sha256(signal.tobytes()).hexdigest()
                    info.update(member=member,bearing=code,label=bi//2,condition=condition,archive=str(archive),
                                mat_sha256=raw_hash,processed_sha256=full_hash,record_index=len(records))
                    records.append(info)
                    group=global_records.get(full_hash)
                    if group is not None and (group["condition"]!=condition or group["bearing"]!=code):
                        raise ValueError(f"Exact processed record duplicates across groups: {member} and {group['member']}")
                    global_records.setdefault(full_hash,info)
                    if full_hash in seen:
                        representative=unique[seen[full_hash]]
                        info["duplicate_of"]=representative["info"]["member"]
                        representative["aliases"].append(member)
                        continue
                    seen[full_hash]=len(unique)
                    unique.append(dict(signal=signal,info=info,aliases=[member]))
                quotient,remainder=divmod(500,len(unique))
                rng=np.random.default_rng(np.random.SeedSequence([SAMPLE_SEED,ci,bi]))
                for ri,record in enumerate(unique):
                    signal,info=record["signal"],record["info"]
                    quota=quotient+int(ri<remainder)
                    candidates=np.arange(0,signal.shape[-1]-LENGTH+1,STRIDE)
                    if len(candidates)<quota:
                        raise ValueError(f"Not enough windows without replacement: {info['member']} {len(candidates)}<{quota}")
                    selected=np.sort(rng.choice(candidates,quota,replace=False))
                    info["selected_windows"]=quota
                    info["candidate_windows"]=len(candidates)
                    if len(record["aliases"])>1:
                        duplicate_groups.append(dict(condition=condition,bearing=code,members=record["aliases"]))
                    record_file=f"r{info['record_index']:04d}.npy"
                    np.save(out/"records"/record_file,signal)
                    info["processed_file"]=record_file
                    identity=int.from_bytes(hashlib.sha256((condition+"/"+info["member"]).encode()).digest()[:4],"little")
                    noise_full=[project_noise(info,seed,identity) for seed in NOISE_SEEDS] if ci else None
                    for start in selected:
                        start=int(start)
                        index=len(windows)
                        x[index]=signal[:,start:start+LENGTH]
                        row=dict(window_index=index,bearing=code,label=bi//2,condition=condition,
                                 role="source" if ci==0 else "target",member=info["member"],record_index=info["record_index"],
                                 start_angular_sample=start,end_angular_sample_exclusive=start+LENGTH,
                                 start_s=info["grid_origin_s"]+info["trim_start_sample"]/RAW_FS+start/(info["fr"]*SPR),
                                 duration_s=LENGTH/(info["fr"]*SPR),noise_record_identity=identity)
                        windows.append(row)
                        if ci:
                            for instance,noise in enumerate(noise_full):
                                templates[index-4000,instance]=noise[start:start+LENGTH]
                del unique
                x.flush()
                templates.flush()
                state.update(prepared_bearing_condition_groups=ci*8+bi+1,prepared_windows=len(windows))
                write_json(out/"config.json",state)
                print("prepared",condition,code,"unique_records",len(seen),"total_windows",len(windows),flush=True)
        if len(windows)!=16000 or len(records)!=640:
            raise ValueError("Incomplete data preparation")
        ids={condition:np.array([r["window_index"] for r in windows if r["condition"]==condition]) for condition in CONDITIONS}
        for condition,indices in ids.items():
            labels=np.array([windows[int(i)]["label"] for i in indices])
            if len(indices)!=4000 or not np.array_equal(np.bincount(labels,minlength=4),np.full(4,1000)):
                raise ValueError("Class sample quota mismatch")
        hashes=defaultdict(list)
        for i in range(len(windows)):
            hashes[hashlib.sha256(x[i].tobytes()).hexdigest()].append(i)
        duplicates=[group for group in hashes.values() if len(group)>1]
        if duplicates:
            raise ValueError(f"Exact duplicate selected windows remain: {duplicates[:8]}")
        x.flush()
        templates.flush()
        del x,templates
        with (out/"indices.npz").open("xb") as f:
            np.savez(f,**ids)
        write_json(out/"windows.json",windows)
        write_json(out/"records.json",records)
        write_json(out/"archive_hashes.json",archive_hashes)
        verification=dict(windows=16000,source_windows=4000,target_windows_each=4000,raw_records=640,
                          unique_processed_records=len(global_records),duplicate_groups=duplicate_groups,
                          exact_duplicate_selected_windows=0,source_target_processed_record_overlap=0,
                          nominal_angular_sampling=SPR,input_length=LENGTH,rotations_per_window=LENGTH/SPR,
                          noise_injection="vibration only before shift/crop/downsampling/order tracking; original 64k record power",
                          noise_instance_seeds=list(NOISE_SEEDS),noise_templates_include_original_vibration_RMS=True,
                          target_descriptors_computed=False,target_model_inference_performed=False,
                          raw_mat_hashes_recorded=True,source_files_sha256={name:sha(PROJECT/name) for name in ("raw_prepare.py","common.py")})
        verification["files_sha256"]={path.name:sha(path) for path in out.iterdir() if path.is_file() and path.name!="config.json"}
        verification["record_files_sha256"]={p.name:sha(p) for p in (out/"records").glob("*.npy")}
        write_json(out/"verification.json",verification)
        state.update(stage="prepared",finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_json(out/"config.json",state)
        print("DATA PREPARED",str(out),flush=True)
    except BaseException:
        state.update(stage="failed",error=traceback.format_exc())
        write_json(out/"config.json",state)
        raise



def reproduce(out):
    """Recreate the corrected cache exclusively from raw archives in a new folder.

    Numeric and sample-identity hashes must match the saved PU-3 input manifest.
    The original cache and frozen results are never overwritten.
    """
    from common import DATA
    if out.resolve() == DATA.resolve():
        raise ValueError('The retained PU-3 input directory cannot be overwritten')
    prepare(out)
    ledger = json.loads((PROJECT/'maintenance/cleanup_20261114/manifest.json').read_text(encoding='utf-8'))
    expected = ledger['data_files_sha256']
    checked = {}
    for name, value in expected.items():
        if name in ('config.json','verification.json','correction_check.json'):
            continue
        if sha(out/name) != value:
            raise ValueError('Recreated PU-3 input differs: '+name)
        checked[name] = value
    proof = json.loads((out/'verification.json').read_text(encoding='utf-8'))
    proof['anti_alias_correction'] = dict(
        source_and_1500rpm_targets_bitwise_unchanged=True,
        corrected_angular_sampling_equivalent_hz=9600,
        additional_resample_ratio='3/5',kaiser_beta=5,
        target_classification_scores_used_for_correction=False,
        reproduced_numeric_and_identity_files=len(checked),
        reference_manifest_sha256=sha(PROJECT/'maintenance/cleanup_20261114/manifest.json'))
    write_json(out/'verification.json',proof)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reproduce(args.output.resolve())
