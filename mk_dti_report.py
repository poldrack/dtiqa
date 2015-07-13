"""
make report for QA using reportlab module
"""

from reportlab.pdfgen import canvas
import numpy as N
import time
import os

def mk_dti_report(infile,dtidir,datavars):
              #imgsnr,meansfnr,spikes,badvols):
    
    timestamp=time.strftime('%B %d, %Y: %H:%M:%S')
    
    report_header=[]
    report_header.append('QA Report: %s'%timestamp)
    report_header.append('directory: %s'%os.path.dirname(infile))
    report_header.append('filename: %s'%os.path.basename(infile))
    report_header.append('Mean SNR: %f'%N.mean(datavars['imgsnr']))
    badvols=['%d'%i for i in datavars['badvols']]
    report_header.append('# potentially bad gradients: %d (%s)'%(len(datavars['badvols']),' '.join(badvols)))
    
    c = canvas.Canvas(os.path.join(dtidir,"QA_report.pdf"))
    yloc=820
    stepsize=16
    for line in report_header:
        c.drawString(10,yloc,line)
        yloc=yloc-stepsize
    
    timeseries_to_draw=['snr.png','fd.png','interleavecorr.png','slicecorr.png']
    
    tsfiles=[os.path.join(dtidir,t) for t in timeseries_to_draw]
    
    ts_img_size=[467,140]
    yloc=yloc-ts_img_size[1]
    
    for imfile in tsfiles:
        c.drawImage(imfile, 45,yloc,width=ts_img_size[0],height=ts_img_size[1])
        yloc=yloc-ts_img_size[1]
    
    c.showPage()
    
#    yloc=650
#    c.drawImage(os.path.join(qadir,'spike.png'),20,yloc,width=500,height=133)
    yloc=330
    images_to_draw=['FA.png','worst_gradient.png']
    imfiles=[os.path.join(dtidir,t) for t in images_to_draw]
    c.drawImage(imfiles[0],0,yloc,width=300,height=300)
    c.drawImage(imfiles[1],300,yloc,width=300,height=300)
#    yloc=20
#    c.drawImage(imfiles[2],0,yloc,width=325,height=325)
    
    c.save()