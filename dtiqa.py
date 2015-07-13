#!/usr/bin/env python
"""
perform QA on DTI data
- based on methods from Liu et al., SPIE 2010

"""

import matplotlib
#avoid triggering X server dependency on compute nodes
matplotlib.use('Agg')
import numpy as N
import nibabel as nib
import os,sys
from run_shell_cmd import *
from compute_fd import *
from plot_timeseries import plot_timeseries
from mk_report import mk_report
import matplotlib.pyplot as plt
from mk_slice_mosaic import *
from mk_dti_report import *

#dtifile='/Users/poldrack/Dropbox/data/dtiqa/DTI/rpo-BOOST-pilot1MRPoldrack_BM8120120821163025ldl58jDTIMB64Dirss008a001.nii.gz'

dtifile=sys.argv[1]

try:
    bet_f=float(sys.argv[2])
except:
    bet_f=0.2

interleavecorr_thresh=0.95

def isodd(num):
            return num & 1 and True or False
        
def compute_disp(motpars):
    # convert rotation to displacement on a 50 mm sphere
    # mcflirt returns rotation in radians
    # from Jonathan Power:
    #The conversion is simple - you just want the length of an arc that a rotational
    # displacement causes at some radius. Circumference is pi*diameter, and we used a 5
    # 0 mm radius. Multiply that circumference by (degrees/360) or (radians/2*pi) to get the
    # length of the arc produced by a rotation.

    headradius=50
    disp=motpars.copy()
    disp=N.abs(disp)
    # note that the order of parameters is differnt here than in mcflirt!
    disp[:,3:6]=N.pi*headradius*2*(disp[:,3:6]/(2*N.pi))

    FD=N.sum(disp,1)

    return FD

def load_ecclog_motion(ecclog_file):
    f=open(ecclog_file)
    l=f.readlines()
    f.close()
    mtx_tex={}
    for lnum in range(len(l)):
        line=l[lnum]
        if line.strip() == '' or line.strip() == 'Final result:':
            continue
        if line.find('processing')==0:
            fname=line.strip().split()[1].split('_ecorr_')[1]
            if not fname.find('tmp0000')>-1:
                imgnum=int(fname.replace('tmp',''))
                mtx_tex[imgnum]=l[(lnum+3):(lnum+7)]
    ngrads=len(mtx_tex)
    mtx=N.zeros((ngrads,4,4))
    for grad in range(ngrads):
        tmp=mtx_tex[grad+1]
        for x in range(4):
            mtx[grad,x,:]=[float(t) for t in tmp[x].strip().split()][0:4]

    motion=N.zeros((ngrads,6))
    for grad in range(ngrads):
        motion[grad,0:3]=mtx[grad,0:3,3]
        # compute  angles for rotation
        motion[grad,3]=N.arctan2(mtx[grad,1,0],mtx[grad,0,0])
        motion[grad,4]=N.arctan2(-1.0*mtx[grad,2,0],N.sqrt(mtx[grad,2,1]**2 + mtx[grad,2,2]**2))
        motion[grad,5]=N.arctan2(mtx[grad,2,1],mtx[grad,2,2])
    return motion
# check for files

dtistub=os.path.basename(dtifile.replace('.nii.gz',''))
bvalfile=dtifile.replace('.nii.gz','.bval')
bvecfile=dtifile.replace('.nii.gz','.bvec')
print 'dtistub: ',dtistub
if not os.path.exists(dtifile):
    print '%s does not exist!'%dtifile
    sys.exit()
    
if not os.path.exists(bvalfile):
    print '%s does not exist!'%bvalfile
    sys.exit()


if not os.path.exists(bvecfile):
    print '%s does not exist!'%bvecfile
    sys.exit()


dtidir=os.path.dirname(dtifile)
if dtidir=='':
    dtidir='./'
dtidir=os.path.abspath(dtidir)

qadir=os.path.join(dtidir,dtistub+'_QA')
if not os.path.exists(qadir):
    os.mkdir(qadir)
    
bvals=N.loadtxt(bvalfile)
bvecs=N.loadtxt(bvecfile).T

# run bet on b0 file
# for this, just grab the first one
# siemens puts small nonzero numbers into the b=0 images
bzero_vols=N.where(bvals<50)[0]
dwi_vols=N.where(bvals>49)[0]
print 'found %d DWI vols and %d b=0 vols'%(len(dwi_vols),len(bzero_vols))
bzero_vol=bzero_vols[0]
bzero_img=dtifile.replace('.nii.gz','_lowb.nii.gz')
if not os.path.exists(bzero_img):
    print 'extracting and betting low-b image'
    cmd='fslroi %s %s %d 1'%(dtifile,bzero_img,bzero_vol)
    print cmd
    run_shell_cmd(cmd)
    cmd='bet %s %s -m -f %f'%(bzero_img,bzero_img.replace('.nii.gz','_brain.nii.gz'),bet_f)
    print cmd
    run_shell_cmd(cmd)
else:
    print 'lowb file already exists'
bzero_mask=bzero_img.replace('.nii.gz','_brain_mask.nii.gz')

# do eddy correction

ecorr_dtifile=dtifile.replace('.nii.gz','_ecorr.nii.gz')
ecorr_log=ecorr_dtifile.replace('.nii.gz','.ecclog')

if not os.path.exists(ecorr_dtifile):
    print 'eddy correcting'
    cmd='eddy_correct %s %s %d'%(dtifile,ecorr_dtifile,bzero_vol)
    print cmd
    run_shell_cmd(cmd)
else:
    print 'eddy corrected file already exists'

motion=load_ecclog_motion(ecorr_log)
motion_file=ecorr_log.replace('ecclog','motpars')
N.savetxt(motion_file,motion)
fd=compute_disp(motion)

# do dtifit

fafile='%s_dtifit_FA.nii.gz'%dtistub
if not os.path.exists(os.path.join(dtidir,fafile)):
    print 'running dtifit'
    cmd='dtifit -k %s -o %s/%s_dtifit -r %s -b %s -m %s -w --sse'%(ecorr_dtifile,dtidir,dtistub,bvecfile,bvalfile,bzero_mask)
    print cmd
    run_shell_cmd(cmd)
else:
    print 'fafile already exists'


# now for the actual QA steps


# check for oblique orientation

# compute slice-wise check for intensity artifacts
# ala Liu et al. (from DTIPrep)

maskimg=nib.load(bzero_mask)
maskdata=maskimg.get_data()
maskvox=N.where(maskdata>0)
nonmaskvox=N.where(maskdata==0)

ecorr_img=nib.load(ecorr_dtifile)
ecorr_data=ecorr_img.get_data()[:,:,:,bvals>50]
xdim,ydim,nslices,ngrads=ecorr_data.shape
print 'doublecheck: found %d gradients'%ngrads
slicecorr=N.zeros((nslices-1,ngrads))
interleavecorr=N.zeros(ngrads)
snr=N.zeros(ngrads)
for grad in range(ngrads):
    for slice in range(nslices-1):
        x=ecorr_data[:,:,slice,grad].reshape(xdim*ydim)
        y=ecorr_data[:,:,slice+1,grad].reshape(xdim*ydim)
        slicecorr[slice,grad]=N.corrcoef(x,y)[0,1]
    tmp=ecorr_data[:,:,:,grad]
    snr[grad]=N.mean(tmp[maskvox])/N.std(tmp[nonmaskvox])
    # now compute odd vs. even slices
    if isodd(nslices):
        oddslices=ecorr_data[:,:,range(0,nslices-1,2),grad]
    else:
        oddslices=ecorr_data[:,:,range(0,nslices,2),grad]
    oddslices=oddslices.reshape(N.prod(oddslices.shape))
    evenslices=ecorr_data[:,:,range(1,nslices,2),grad]
    evenslices=evenslices.reshape(N.prod(evenslices.shape))
    interleavecorr[grad]=N.corrcoef(oddslices,evenslices)[0,1]

worst_gradient=N.where(interleavecorr==N.min(interleavecorr))[0]
fig=plt.figure(figsize=[10,3])
fig.subplots_adjust(bottom=0.15)
plt.plot(slicecorr)
plt.xlabel('slices')
plt.ylabel('correlation between adjacent slices')
plt.title('Slice-wise correlation - lines reflect different gradients')
plt.savefig(os.path.join(qadir,'slicecorr.png'),bbox_inches='tight')
plt.close()

# make report
plot=1
if plot:
    plot_timeseries(fd,'Framewise displacement',os.path.join(qadir,'fd.png'),
        ylabel='FD',xlabel='Gradient directions')
    plot_timeseries(snr,'SNR',os.path.join(qadir,'snr.png'),
        ylabel='SNR',xlabel='Gradient directions')
    plot_timeseries(interleavecorr,'Interleave correlation',os.path.join(qadir,'interleavecorr.png'),
        ylabel='Correlation between odd/even slices',xlabel='Gradient directions')


fa_img=nib.load(os.path.join(dtidir,fafile))
fa_data=fa_img.get_data()
mk_slice_mosaic(fa_data,os.path.join(qadir,'FA.png'),'FA (with mask)',contourdata=maskdata)
sse_img=nib.load(os.path.join(dtidir,fafile.replace('FA','sse')))
sse_data=sse_img.get_data()
mk_slice_mosaic(sse_data,os.path.join(qadir,'sse.png'),'SSE of dtifit')

mk_slice_mosaic(ecorr_data[:,:,:,worst_gradient],
    os.path.join(qadir,'worst_gradient.png'),
    'worst gradient (%d: intcorr=%0.2f)'%(worst_gradient,interleavecorr[worst_gradient]))

# spit out the info about the scan

#qafile=open(os.path.join(dtifile.replace('.nii.gz','.QAreport')),'w')
#qafile.write('DTI QA Report\n')
#qafile.write('Directory: %s\n'%dtidir)
#qafile.write('File: %s\n'%dtifile)

# check for bad gradients
bad_grads=N.where(interleavecorr<interleavecorr_thresh)[0]
if len(bad_grads)>0:
    print 'Possibly bad gradients:'
    for i in bad_grads:
        print '%d: %f'%(i,interleavecorr[i])
else:
    print 'No bad gradients (by interleave correlation > %f)'%interleavecorr_thresh

#qafile.close()

datavars={'imgsnr':snr,'badvols':bad_grads}
mk_dti_report(dtifile,qadir,datavars)

