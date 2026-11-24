"""仅来源数据参与训练与选模，普通CE、逐epoch验证及早停。"""
import torch
import time
import numpy as np
import pandas as pd
from transfer_split import selected


def train_case(R, x, windows, case, destination, max_epochs, device, patience=20):
    if max_epochs < 1 or patience < 1:
        raise ValueError('max_epochs和patience必须为正整数')
    protocol, method, ratio, subset, seed = case
    R.U.seed_all(seed)
    ti = selected(windows, ratio, subset)
    vi = np.flatnonzero(windows.role.eq('validation'))
    y = windows.label.to_numpy()
    rpm = windows.rpm.to_numpy(np.float32)
    weights = R.U.weights_for(windows, vi)
    model = R.Classifier(method, R.U.CFG).to(device)
    R.U.sync(); start = time.perf_counter(); fitting_start = start
    model.fit(torch.as_tensor(x[ti], device=device), torch.as_tensor(rpm[ti], device=device))
    R.U.sync(); fit_seconds = time.perf_counter()-start
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs, eta_min=.00001)
    validation_views = list(R.conditions(x[vi], protocol, False))
    destination.mkdir(parents=True)
    windows.iloc[ti].to_csv(destination/'selected_windows.csv', index=False)
    best, best_update, step, epoch = float('inf'), 0, 0, 0
    best_epoch, stale_epochs = 0, 0
    validation_seconds = 0.; validation_calls = 0; history = []
    while epoch < max_epochs:
        epoch += 1
        rng = np.random.default_rng(np.random.SeedSequence([subset, seed, epoch, 20261011]))
        order = R.U.sample_epoch(windows, ti, rng)
        loss_sum = 0.; batches = 0; seconds = 0.; score = np.nan
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for offset in range(0, len(order), 32):
            model.train(); R.U.sync(); start = time.perf_counter()
            indices = order[offset:offset+32]
            observed = x[indices]
            if protocol == 'awgn_mixed':
                rng = np.random.default_rng(np.random.SeedSequence([20261012, subset, seed, epoch, offset]))
                observed = R.observe(observed, rng)[0]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(torch.as_tensor(observed, device=device),
                                  torch.as_tensor(rpm[indices], device=device))
            loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(y[indices], device=device))
            if not torch.isfinite(loss):
                raise ValueError('训练损失非有限值')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()); batches += 1; step += 1
            R.U.sync(); seconds += time.perf_counter()-start
        peak = torch.cuda.max_memory_allocated()/1024**2 if device.type == 'cuda' else None
        start = time.perf_counter()
        values = [R.U.nll(R.predict(model, wave, rpm[vi], device), y[vi], weights)
                  for _, _, wave in validation_views]
        R.U.sync(); validation_seconds += time.perf_counter()-start
        validation_calls += len(validation_views)
        score = values[0] if protocol == 'clean_only' else .5*values[0]+.1*sum(values[1:])
        if not np.isfinite(score):
            raise ValueError('验证评分非有限值')
        if score < best-1e-6:
            best, best_update, best_epoch = score, step, epoch
            stale_epochs = 0
            torch.save(dict(method=method, model_config=R.U.CFG,
                            state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),
                       destination/'model.pt')
        else:
            stale_epochs += 1
        print(f'{case}: epoch={epoch}, updates={step}, validation_NLL={score:.5f}, stale={stale_epochs}/{patience}', flush=True)
        with (destination/'validation.csv').open('a', encoding='utf-8') as handle:
            if epoch == 1:
                handle.write('epoch,updates,validation_nll,stale_epochs\n')
            handle.write(f'{epoch},{step},{score:.17g},{stale_epochs}\n')
        scheduler.step()
        history.append(dict(epoch=epoch, updates=step, train_loss=loss_sum/batches,
                            validation_nll=score, train_seconds=seconds, peak_gpu_mib=peak))
        if stale_epochs >= patience:
            break
    history = pd.DataFrame(history)
    history.to_csv(destination/'training.csv', index=False)
    R.learning_curve(history, destination/'training.png')
    info = dict(protocol=protocol, method=method, ratio=ratio, subset=subset, seed=seed,
                best_update=best_update, updates_trained=step, validation_nll=best,
                best_epoch=best_epoch, epochs_trained=epoch, max_epochs=max_epochs,
                early_stopping_patience=patience, early_stopped=stale_epochs >= patience,
                stopping_reason='patience' if stale_epochs >= patience else 'max_epochs',
                loss='cross_entropy', label_smoothing=0.0, validation_interval_epochs=1,
                selection_min_delta=1e-6, scheduler_interval='epoch',
                training_windows=len(ti), validation_windows=len(vi), from_scratch=True,
                training_class_support=windows.iloc[ti].groupby('label').size().tolist(),
                training_records=int(windows.iloc[ti].record_id.nunique()),
                train_test_shared_records=len(set(windows.iloc[ti].record_id) &
                                              set(windows.loc[windows.role.eq('test'), 'record_id'])),
                historical_test_windows_now_training=int(windows.iloc[ti].historical_role.eq('test').sum()),
                fit_seconds=fit_seconds, parameters=sum(p.numel() for p in model.parameters()),
                train_seconds_per_epoch=float(history.train_seconds.mean()),
                total_train_seconds=float(history.train_seconds.sum()),
                peak_gpu_mib=float(history.peak_gpu_mib.max()) if device.type == 'cuda' else None,
                validation_forward_seconds=validation_seconds, validation_forward_calls=validation_calls,
                fit_train_validation_save_seconds=time.perf_counter()-fitting_start)
    R.write_json(destination/'parameters.json', info)
    return info
