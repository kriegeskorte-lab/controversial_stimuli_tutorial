import argparse

import itertools
import random
import requests
import pathlib, os
import time

import torch
import lucent.optvis as ov
import lucent.optvis.transform
import lucent.optvis.param

import models
from optimize import optimize_controversial_stimuli, optimize_controversial_stimuli_with_lucent, make_GANparam


def cluster_check_if_exists(target_fpath,max_synthesis_time=60*10,verbose=True):
    """ returns true if target_fpath does NOT exist and there are no associated flag files.
    This is a simple way to coordinate multiple processes/nodes through a shared filesystem (e.g., as in SLURM).

    Either one of two flag files may cause this function to return True:
        [target_fpath].in_process_flag, which communicates that another worker is now working on this file.
        [target_fpath].failed_flag, which communicates that previous attempt at this file has failed.
    Args:
        target_fpath (str): the filename to be produced.
        max_synthesis_time (int/float): delete in-process file if it is older than this number of seconds.
        verbose (boolean)
    """

    file=pathlib.Path(target_fpath)
    verboseprint = print if verbose else lambda *a, **k: None # https://stackoverflow.com/a/5980173

    # make sure target folder is there
    assert file.parent.exists(), str(file.parent) + " must exist and be reachable."

    if file.exists():
        verboseprint(target_fpath+" found, skipping.")
        return True

    in_process_flag=pathlib.Path(target_fpath + '.in_process_flag')

    if in_process_flag.exists(): # check if other cluster worker is currently optimizing this file
        # but is it too old?
        flag_age=time.time()-in_process_flag.stat().st_mtime
        if flag_age < max_synthesis_time: # no, it's fresh.
            verboseprint('a fresh '+ str(in_process_flag)+ " found, skipping.")
            return True
        else: # old flag. might be have been left by killed workers.
            verboseprint(str(in_process_flag)+ " found, but it is old.")
            try:
                in_process_flag.unlink()
            except:
                pass
            # now, try again.
            return cluster_check_if_exists(target_fpath,max_synthesis_time=max_synthesis_time,verbose=verbose)
    else: # no in-process flag.
        # wait a little bit to avoid racing with other cluster workers that might have started in the same time.
        random.seed()
        time.sleep(random.uniform(0,1))
        if in_process_flag.exists():
            verboseprint(str(in_process_flag) + " appeared after double checking, skipping.")
            return True
        try:
            in_process_flag.touch(mode=0o777, exist_ok=False)
        except:
            return True
        return False

def cluster_check_if_failed(target_fpath):
    failure_flag=pathlib.Path(target_fpath + '.failed_flag')
    return failure_flag.exists()

def remove_in_process_flag(target_fpath):
    in_process_flag=pathlib.Path(target_fpath + '.in_process_flag')
    in_process_flag.unlink()

def leave_failure_flag(target_fpath):
    failure_flag=pathlib.Path(target_fpath + '.failed_flag')
    failure_flag.touch(mode=0o777, exist_ok=False)

def prepare_optimization_parameters(optim_method):
    """ return preset optimization parameters"""

    optimization_kwd={'smooth_min_kind':'logsoftmax',
               'pytorch_optimizer':'Adam',
               'optimizer_kwargs':{'lr':5e-2,'betas':(0.9, 0.999),'weight_decay':0,'eps':1e-8},
               'return_PIL_images':True,
               'verbose':True}

    if optim_method=='direct':
        optim_func=optimize_controversial_stimuli
        optimization_kwd.update({'im_size':(1,3,256,256)})
    else: # indirect optimization, define param_f and transforms
        optim_func=optimize_controversial_stimuli_with_lucent
        if optim_method=='jittered':
            param_f= lambda: ov.param.image(w=256, h=256, batch=1, decorrelate=False,fft=False, sd=1) # this creates a pixel representation and an initial image that are both similar to those we have used in part 1.
            transforms=[ov.transform.jitter(25)] # a considerable spatial jitter. use transforms=[] for no transforms.
        elif optim_method=='decorrelated':
            param_f= lambda: ov.param.image(w=256, h=256, batch=1, decorrelate=True,fft=True)
            transforms=[ov.transform.jitter(25)]
        elif optim_method=='CPPN':
            param_f = lambda: ov.param.cppn(256)
            transforms=[ov.transform.jitter(5)]
            optimization_kwd['optimizer_kwargs'].update({'lr':5e-3,'eps':1e-3}) # CPPN specific Adam parameters.

        elif optim_method in ['GAN-pool5','GAN-fc6','GAN-fc7','GAN-fc8']:
            GAN_latent_layer=optim_method.split('-')[-1]
            param_f=make_GANparam(batch=1, sd=1,latent_layer=GAN_latent_layer)
            transforms=[ov.transform.jitter(5)]
        else:
            raise ValueError('Unknown optim_method '+optim_method)
        optimization_kwd.update({'param_f':param_f,'transforms':transforms})
    return optim_func, optimization_kwd

def design_synthesis_experiment(exp_name):
        # build particular model and class combinations for the tutorial figures.

        # get imagenet categories
        imagenet_dict_url='https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        class_dict=eval(requests.get(imagenet_dict_url).text)

        if exp_name=='8_random_classes':
            model_pairs=[['Resnet_50_l2_eps5','Wide_Resnet50_2_l2_eps5']]
            n_classes=8
            all_class_idx=list(class_dict.keys())
            random.seed(1)
            class_idx=list(random.sample(all_class_idx,n_classes))
            class_idx_pairs=itertools.product(class_idx,repeat=2)
        elif exp_name=='cat_vs_dog':
            model_names=['Resnet50','InceptionV3','Resnet_50_l2_eps5','Wide_Resnet50_2_l2_eps5']
            model_pairs=itertools.product(model_names,repeat=2)
            model_pairs=[pair for pair in model_pairs if pair[0]!=pair[1]]
            class_idx_pairs=[[283,178]]

        #indecis to classes
        class_pairs=[[class_dict[idx] for idx in idx_pair] for idx_pair in class_idx_pairs]
        return model_pairs, class_pairs

def batch_optimize(target_folder,model_pairs,class_pairs,optim_method,min_controversiality=0.85,max_seeds_to_try=5,max_steps=1000, verbose=True):
    """ Synthesize a batch of controversial stimuli.
    For each model pair, synthesizes a controversial stimuli for all class pairs.

    Args:
    target_folder (str): Where the image are saved.
    model_pairs (list): A list of tuples, each tuple containing two model names.
    class_pairs (list): A list of tuples, each tuple containing two class names.
    optim_method (str): direct/jittered/decorrelated/CPPN/'GAN-pool5'/'GAN-fc6'/'GAN-fc7'/'GAN-fc8'
    min_controversiality (float): minimum controversiality required for saving an image (e.g., 0.85).
    seed (int): fixed random seed or None.

    returns True if one or more synthesized images was not save due to insufficient controversiality.
    """

    verboseprint = print if verbose else lambda *a, **k: None # https://stackoverflow.com/a/5980173

    if torch.cuda.device_count()>1:
        model_1_device='cuda:0'
        model_2_device='cuda:1'
        print('using two GPUs, one per model.')
    elif torch.cuda.device_count()==1:
        model_1_device=model_2_device='cuda:0'
        print('using one GPU.')
    else:
        model_1_device=model_2_device='cpu'
        print('using CPU')

    pathlib.Path(target_folder).mkdir(parents=True, exist_ok=True)
    at_least_one_synthesis_failed=False

    for model_1_name, model_2_name in model_pairs:
        models_loaded=False
        for class_pair in class_pairs:
            class_1_name, class_2_name = class_pair
            # build filename
            short_class_1_name=class_1_name.split(',',1)[0].replace(' ','_')
            short_class_2_name=class_2_name.split(',',1)[0].replace(' ','_')

            # check which seed to optimize
            seed=0
            while seed<max_seeds_to_try:
                target_fname='{}-{}_vs_{}-{}_seed{}.png'.format(model_1_name,short_class_1_name,model_2_name,short_class_2_name,seed)
                target_fpath=os.path.join(target_folder,target_fname)
                if cluster_check_if_failed(target_fpath):
                    seed+=1
                else: # no failure flag file found
                    break
            # give up if already failed max_seeds_to_try times
            if seed==max_seeds_to_try:
                verboseprint(target_fpath + " max seeds tried.")
                continue

            # check if png file already exists and leave 'in-process' flag
            if cluster_check_if_exists(target_fpath):
                verboseprint("Skipping "+target_fpath)
                continue

            print('Synthesizing '+target_fpath)
            if not models_loaded:
                model_1=getattr(models,model_1_name)()
                model_2=getattr(models,model_2_name)()
                model_1.load(model_1_device)
                model_2.load(model_2_device)
                models_loaded=True

            optim_func, optimization_kwd=prepare_optimization_parameters(optim_method)
            optimization_kwd['random_seed']=seed
            optimization_kwd['max_steps']=max_steps
            # run optimization
            _,PIL_ims,controversiality_scores=optim_func(model_1,model_2,class_1_name,class_2_name,**optimization_kwd)
            remove_in_process_flag(target_fpath)

            if class_1_name == class_2_name or controversiality_scores[0]>=min_controversiality:
                PIL_ims[0].save(target_fpath)
                print('saved '+target_fpath)
            else:
                print('insufficient controversiality:',target_fpath,controversiality_scores[0],'not saving file.')
                at_least_one_synthesis_failed=True
                leave_failure_flag(target_fpath) # leave a flag so we don't try again this seed.
    return at_least_one_synthesis_failed

def grand_batch(experiments,optimization_methods,target_folder='optimization_results',min_controversiality=0.85,max_steps=1000):

    task_list=[]
    for exp_name in experiments:
        for optim_method in optimization_methods:
            target_subfolder=os.path.join(target_folder,optim_method+'_optim_'+exp_name)
            task_list.append({'target_subfolder':target_subfolder,
                             'exp_name':exp_name,
                             'optim_method':optim_method})
    while len(task_list)>0:
        cur_task=task_list.pop(0)
        model_pairs, class_pairs=design_synthesis_experiment(cur_task['exp_name'])
        at_least_one_synthesis_failed=batch_optimize(cur_task['target_subfolder'],
            model_pairs,class_pairs,cur_task['optim_method'],
            min_controversiality=min_controversiality,max_seeds_to_try=5,max_steps=max_steps)
        if at_least_one_synthesis_failed:
            task_list.append(cur_task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='batch_optimize',
        epilog='To prepare images for the first controversial stimuli matrix in the tutorial, run "batch_optimize --experiments cat_vs_dog --optimization_methods direct". To prepare all images for all figures, run without --experiments and --optimization_methods arguments. This requires a cluster, or a lot of patience. This program can be run in parallel by multiple nodes sharing a filesystem (as in SLURM).')
    parser.add_argument(
        "--experiments", nargs='+',
        choices= ['cat_vs_dog', '8_random_classes'],
        default= ['cat_vs_dog', '8_random_classes'],
        required=False)
    parser.add_argument(
        "--optimization_methods",nargs='+',
        choices= ['direct','jittered','decorrelated','CPPN','GAN-pool5','GAN-fc6','GAN-fc7','GAN-fc8'],
        default= ['direct','jittered','decorrelated','CPPN','GAN-pool5','GAN-fc6','GAN-fc7','GAN-fc8'])
    parser.add_argument(
        "--target_folder",nargs=1,default="optimization_results")
    parser.add_argument(
        "--max_steps",nargs=1,default=1000)
    parser.add_argument(
        "--min_controversiality",nargs=1,default=0.85)
    args=parser.parse_args()

    grand_batch(**vars(args))