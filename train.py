import os
from datetime import datetime
from re import sub, subn
import copy

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch
import torch.utils.data
import torch.cuda
import torchaudio
import torchvision
import torchsummary
import wandb

from tqdm import tqdm
import matplotlib.pyplot as plt
from random_words import RandomWords

from evaluate import evaluate
from hppnet import *

rw = RandomWords()
random_word_str = rw.random_word()
time_str = datetime.now().strftime('%y%m%d-%H%M%S') + '_' + random_word_str
ex = Experiment('train_transcriber')
ex.time_str = time_str

# Optional MongoDB observer. Disabled by default so training does not depend on a
# reachable Mongo host; set HPPNET_MONGO=<host:port> to enable it.
if os.environ.get('HPPNET_MONGO'):
    try:
        mongo_ob = MongoObserver.create(url=os.environ['HPPNET_MONGO'], db_name='piano_transcription') #harmonic_net_mono
        ex.observers.append(mongo_ob)
    except Exception as e:
        print(f'[warn] Mongo observer disabled: {e}')

ex.tags = []





@ex.config
def config():
    logdir = 'runs/transcriber-' + time_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 600*1000
    resume_iteration = None
    checkpoint_interval = 2000
    train_on = 'MAESTRO'

    batch_size = 3
    sequence_length = 327680
    model_complexity = 48

    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 400

    # How often (in steps) to push training-loss scalars to wandb.
    log_interval = 10

    test_interval= None

    
    ex.observers.append(FileStorageObserver.create(logdir))


    training_size = 1.0 # [1.0, 0.3, 0.1] preportion used for training in training set.

    notes = ""


@ex.config
def model_config():
    SUBNETS_TO_TRAIN = ['onset_subnet', 'frame_subnet'] 
    onset_subnet_heads = ['onset']
    frame_subnet_heads = ['frame', 'offset', 'velocity']

    model_name = "HPPNet"

    fixed_dilation = 24
    model_size = 128

    # Sequence model inside the frequency-grouped heads (ablation).
    #   seq_model:  'lstm' (baseline) | 'mamba' (causal) | 'bimamba' (bidirectional)
    #   mamba_impl: 'mamba1' | 'mamba2'   (only used when seq_model != 'lstm')
    seq_model = 'lstm'
    mamba_impl = 'mamba1'

    # Trunk / "acoustic model" (ablation).
    #   trunk: 'cnn' (baseline harmonic dilated convs) | 'patch' (AuM/ViT-style patch
    #          embedding + sequence-over-frequency, using the same seq_model above).
    #   patch_trunk_depth: number of stacked sequence blocks in the patch trunk.
    trunk = 'cnn'
    patch_trunk_depth = 2


@ex.named_config
def hpp_base():
    model_size = 128
    SUBNETS_TO_TRAIN = ['onset_subnet']
    onset_subnet_heads = ['onset','frame', 'offset', 'velocity']
    frame_subnet_heads = []
    batch_size=4
    iterations = 600*1000

@ex.named_config
def hpp_tiny():
    model_size = 64
    SUBNETS_TO_TRAIN = ['onset_subnet']
    onset_subnet_heads = ['onset','frame', 'offset', 'velocity']
    frame_subnet_heads = []
    batch_size=4
    iterations = 600*1000

@ex.named_config
def hpp_ultra_tiny():
    model_size = 48
    SUBNETS_TO_TRAIN = ['onset_subnet']
    onset_subnet_heads = ['onset','frame', 'offset', 'velocity']
    frame_subnet_heads = []
    batch_size=4
    iterations = 600*1000


# ---------------------------------------------------------------------------
# Sequence-model ablation: replace the frequency-grouped BiLSTM with a
# (bi)directional Mamba SSM. Combine with any model config, e.g.
#   python train.py with hpp_base bimamba
# Requires a CUDA GPU and: pip install --no-build-isolation causal-conv1d mamba-ssm
# ---------------------------------------------------------------------------
@ex.named_config
def mamba():
    seq_model = 'mamba'
    mamba_impl = 'mamba1'
    model_name = 'HPPNet-Mamba'

@ex.named_config
def bimamba():
    seq_model = 'bimamba'
    mamba_impl = 'mamba1'
    model_name = 'HPPNet-BiMamba'

@ex.named_config
def mamba2():
    seq_model = 'mamba'
    mamba_impl = 'mamba2'
    model_name = 'HPPNet-Mamba2'

@ex.named_config
def bimamba2():
    seq_model = 'bimamba'
    mamba_impl = 'mamba2'
    model_name = 'HPPNet-BiMamba2'


# ---------------------------------------------------------------------------
# Trunk ablation: replace the harmonic dilated conv acoustic model with an
# AuM/ViT-style patch-embedding trunk that patchifies the CQT and runs a
# sequence model over the frequency axis. Orthogonal to the seq-model configs,
# so it composes with any size + seq-model config, e.g.
#   python train.py with hpp_tiny patchify            # patch trunk + LSTM
#   python train.py with hpp_tiny patchify bimamba    # patch trunk + BiMamba
# (patchify+mamba/bimamba requires a CUDA GPU, like the seq-model ablation.)
# ---------------------------------------------------------------------------
@ex.named_config
def patchify():
    trunk = 'patch'
    model_name = 'HPPNet-Patch'


@ex.config
def loss_config():
    positive_weight = 2
    
@ex.config
def train_without_test():
    test_interval = None
    test_onset_threshold = 0.4
    test_frame_threshold = 0.4

@ex.named_config
def train_with_test():
    # validation_interval = 50
    test_interval = 500000

    test_onset_threshold = 0.4
    test_frame_threshold = 0.4



ex.main_locals = locals()


def init_wandb(logdir, config):
    """Start (or resume) the Weights & Biases run for this job.

    Project / run name / group / id / resume are read from the standard wandb env vars
    (WANDB_PROJECT, WANDB_NAME, WANDB_RUN_GROUP, WANDB_RUN_ID, WANDB_RESUME, WANDB_JOB_TYPE)
    when set by scripts/runpod_train_eval.sh; otherwise sensible defaults are derived here.
    Best-effort: if wandb can't start (e.g. no WANDB_API_KEY) fall back to a disabled run so
    training still proceeds.
    """
    kwargs = dict(
        project=os.environ.get('WANDB_PROJECT', 'hppnet-mamba-ablation'),
        name=os.environ.get('WANDB_NAME', config.get('model_name', 'HPPNet') + '_' + time_str),
        group=os.environ.get('WANDB_RUN_GROUP'),
        job_type=os.environ.get('WANDB_JOB_TYPE', 'train'),
        config=config,
        dir=logdir,
    )
    run_id = os.environ.get('WANDB_RUN_ID')
    if run_id:
        kwargs['id'] = run_id
        kwargs['resume'] = os.environ.get('WANDB_RESUME', 'allow')
    try:
        return wandb.init(**kwargs)
    except Exception as e:
        print(f'[warn] wandb disabled ({e}); logging to a local disabled run')
        return wandb.init(mode='disabled', config=config)


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
           learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval, 
          test_interval, test_onset_threshold, test_frame_threshold, 
          training_size,
          model_name):
    print_config(ex.current_run)

    config = ex.current_run.config

    SUBNETS_TO_TRAIN = config['SUBNETS_TO_TRAIN']


    # add source files to ex
    # Locate the FileStorageObserver by type (its list index depends on whether the
    # optional Mongo observer is enabled).
    fs_observer = next(o for o in ex.current_run.observers if isinstance(o, FileStorageObserver))
    src_file_dir = os.path.join(fs_observer.dir, 'src')
    # Best-effort source snapshot for provenance; never let it break training.
    try:
        src_file_set = set()
        utils.save_src_files(ex.main_locals, src_file_dir, query_str='hppnet', src_path_set=src_file_set)
        for src_path in src_file_set:
            if os.path.exists(src_path):
                ex.add_source_file(src_path)
    except Exception as e:
        print(f'[warn] source snapshot skipped: {e}')

    utils.copy_dir('./', src_file_dir)
    utils.copy_dir('./hppnet', os.path.join(src_file_dir, 'hppnet'))


    os.makedirs(logdir, exist_ok=True)
    init_wandb(logdir, dict(config))
    log_interval = config['log_interval']

    ex.basedir = fs_observer.basedir

    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    if train_on == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length)
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
        
        # test
        test_dataset = MAESTRO(groups=['test'])
        
        # groups = test_dataset.groups
        # test_dataset = torch.utils.data.Subset(test_dataset, list(range(3)))
        # test_dataset.groups = groups
    else:
        dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'], sequence_length=sequence_length)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length)
        test_dataset = MAPS(groups=[['ENSTDkAm', 'ENSTDkCl']])
        # test_dataset = MAESTRO(groups=['test'])

    train_idx = [int(x/training_size) for x in range(int(len(dataset)*training_size))]
    ex.info['training_files'] = dataset.files('train')
    ex.info['training_idx'] = train_idx
    dataset = torch.utils.data.Subset(dataset, train_idx)
    ex.info['train_num'] = len(dataset) 

    ex.info['validation_set_files'] = validation_dataset.files('validation')
    ex.info['test_set_files'] = test_dataset.files('test')

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=4)

    # validation_dataset = DataLoader(validation_dataset, num_workers=4)


    optimizers = {}
    if resume_iteration is None:
        model = HPPNet(N_MELS, MAX_MIDI - MIN_MIDI + 1, config).to(device)

        # optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        for subnet in SUBNETS_TO_TRAIN:
            optimizers[subnet] = torch.optim.Adam(model.subnets[subnet].parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        # optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        # optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))
        for subnet in SUBNETS_TO_TRAIN:
            optimizers[subnet] = torch.optim.Adam(model.subnets[subnet].parameters(), learning_rate)
            optimizers[subnet].load_state_dict(torch.load(os.path.join(logdir, f'last-optimizer-state-{subnet}.pt')))
            
    # summary
    # torchsummary.summary(model, input_size=(1, 16000*4, ), batch_size=1, device='cpu')
    # writer.add_graph(model, torch.zeros([2, 16000*20]))
    # summary(model)
    summary_path = ex.basedir + '/model_summary.txt'
    summary(model, summary_path)
    ex.add_artifact(summary_path)

    # scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)
    schedulers = {}
    for subnet in SUBNETS_TO_TRAIN:
        schedulers[subnet] = StepLR(optimizers[subnet], step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)
    

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    loop.set_description(config['model_name'] + '_' + random_word_str)
    tqdm_dict = {}
    for i, batch in zip(loop, cycle(loader)):

        batch['audio'] = batch['audio'].to(device)
        batch['onset'] = batch['onset'].to(device)
        batch['offset'] = batch['offset'].to(device)
        batch['frame'] = batch['frame'].to(device)
        batch['velocity'] = batch['velocity'].to(device)



        predictions, losses = model.run_on_batch(batch)

        loss = sum(losses.values())

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # scheduler.step()

        for subnet in SUBNETS_TO_TRAIN:
            loss_subnet = losses[f'loss/{subnet}']
            optimizers[subnet].zero_grad()
            loss_subnet.backward()
            optimizers[subnet].step()
            schedulers[subnet].step()

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        if i % log_interval == 0:
            wandb.log({'train/' + key: value.item() for key, value in {'loss': loss, **losses}.items()}, step=i)

        if(i %10 == 0):
            tqdm_dict['train/loss'] = loss.cpu().detach().numpy()
            loop.set_postfix(tqdm_dict)

        if(i in [100, 1000, 2000, 4000, 8000] or i % 10000 == 0 ):
            frame_img_pred = torch.swapdims(predictions['frame'], 1, 2)
            frame_img_pred = torch.unsqueeze(frame_img_pred, dim=1)
            # => [F x T]
            frame_img_pred = torchvision.utils.make_grid(frame_img_pred, pad_value=0.5)
            # writer.add_image('train/step_%d_pred'%i, frame_img_pred)

            frame_img_ref = torch.swapdims(batch['frame'], 1, 2)
            frame_img_ref = torch.unsqueeze(frame_img_ref, dim=1)
            frame_img_ref = torchvision.utils.make_grid(frame_img_ref, pad_value=0.5)
            # writer.add_image('train/step_%d_ref'%i, frame_img_ref)

            frame_img = torch.cat([frame_img_ref[0], frame_img_pred[0]], dim=0)
            dir_path = os.path.join(logdir, 'piano_roll')
            os.makedirs(dir_path, exist_ok=True)
            plt.imsave(dir_path + '/train_step_%d.png'%(i), frame_img.detach().cpu().numpy())
            wandb.log({'piano_roll/ref_vs_pred': wandb.Image(frame_img.detach().cpu().numpy())}, step=i)

        ##################################
        # Validate
        if i % validation_interval == 0:
            print("validating...")
            model.eval()
            with torch.no_grad():
                val_metrics = evaluate(validation_dataset, model, device)
                val_log = {}
                for key, value in val_metrics.items():
                    mean_val = torch.tensor(value).cpu().numpy().mean()
                    label = 'validation/' + key.replace(' ', '_')
                    val_log[label] = mean_val
                    ex.log_scalar(label, mean_val, i)
                wandb.log(val_log, step=i)
                # tqdm_dict['on_loss'] = '%.4f'%np.mean(val_metrics['loss/onset'])
                tqdm_dict['f_f1'] = '%.3f'%np.mean(val_metrics['metric/frame/f1'])
                tqdm_dict['n_f1'] = '%.3f'%np.mean(val_metrics['metric/note/f1'])
                loop.set_postfix(tqdm_dict)
            model.train()

        ##################################
        # Test
        if not test_interval is None:
            if i % test_interval == 0:
                print("testing...")
                model.eval()
                clip_len = 10240
                test_result = {}
                test_result['step'] = i
                test_result['time'] = datetime.now().strftime('%y%m%d-%H%M%S')
                test_result['dataset'] = str(test_dataset)
                test_result['dataset_group'] = test_dataset.groups
                test_result['dataset_len'] = len(test_dataset)
                test_result['clip_len'] = clip_len
                test_result['onset_threshold'] = test_onset_threshold
                test_result['frame_threshold'] = test_frame_threshold
                with torch.no_grad():
                    eval_result =  evaluate(test_dataset, model, device,
                        onset_threshold=test_onset_threshold, frame_threshold=test_frame_threshold,
                        clip_len = clip_len,
                        save_path=config['logdir'] + f'/model-{i}-test'
                    )
                    wandb_test_log = {}
                    for key, values in eval_result.items():
                        mean_val = np.mean(values)
                        # std_val = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
                        label = 'test/' + key.replace(' ', '_')
                        ex.log_scalar(label, mean_val, i)
                        test_result[label] = "%.2f"%(mean_val*100)
                        if key.startswith('metric/'):
                            wandb_test_log['test/' + key[len('metric/'):]] = float(mean_val)
                    wandb.log(wandb_test_log, step=i)
                    wandb.summary.update(wandb_test_log)
                ex.info[f'test_step_{i}'] = test_result
                model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))

            # torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
            for subnet in SUBNETS_TO_TRAIN:
                torch.save(optimizers[subnet].state_dict(), os.path.join(logdir, f'last-optimizer-state-{subnet}.pt'))

    wandb.finish()
