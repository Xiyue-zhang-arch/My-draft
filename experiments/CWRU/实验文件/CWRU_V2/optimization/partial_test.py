"""等待用户指定的两系数暂停边界，然后测试一次，保持训练暂停。"""
from pathlib import Path
import sys
import argparse
import json
import time
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optimization.runner import Suite, checked_root, write_json, now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    root=checked_root(args.output)
    request=root/'partial_test_request.json'
    if request.exists():
        raise FileExistsError('已存在本次提前测试请求，禁止重复启动')
    state=dict(status='waiting_for_pause',requested_at=now(),coefficients=[.01,.03],
               scope='全部72新模型完成后，与36历史数据对照一起测试；不继续训练')
    write_json(request,state)
    print('WAITING FOR COMPLETE LAMBDA 0.01/0.03 AND VERIFIED TRAINING PAUSE',flush=True)
    try:
        while True:
            try:
                current=json.loads((root/'config.json').read_text(encoding='utf-8'))
            except (json.JSONDecodeError,PermissionError):
                time.sleep(1)
                continue
            if current['stage']=='paused' and current.get('pause_boundary_verified'):
                break
            if current['stage']!='lambda':
                raise RuntimeError('未到指定暂停边界或训练异常，保留现场，禁止自动测试')
            pause=json.loads((root/'pause_request.json').read_text(encoding='utf-8'))
            if pause['status'] not in ('watching','paused'):
                raise RuntimeError('暂停控制未正常推进，禁止自动测试')
            time.sleep(1)
        state.update(status='testing',started_at=now())
        write_json(request,state)
        Suite(root).evaluate(partial=True)
        state.update(status='complete',completed_at=now(),training_remains_paused=True)
        write_json(request,state)
        print('PARTIAL TEST COMPLETE; TRAINING REMAINS PAUSED',flush=True)
    except BaseException as error:
        state.update(status='failed',error=str(error),updated_at=now())
        write_json(request,state)
        raise


if __name__=='__main__':
    main()
