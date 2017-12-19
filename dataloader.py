import cv2
import numpy as np
import  os
import tensorflow as tf
import time
import threading
import Queue
import random
mode='npy'
#unfinished for mode dir
#my own implement of data queue
class dataloader(threading.Thread):
    def __init__(self,datapath,datalistfile,name,bs):
        self.datapath=datapath
        threading.Thread.__init__(self, name=name)
        self.datalist=open(datalistfile).readlines()
        self.dataqueue=Queue.Queue(maxsize=10)
        self.length=len(self.datalist)
        self.on=True
        self.bs=bs
    def run(self):
        while(self.on):
            data=[]
            for i in xrange(self.bs): 
                k=random.randint(0,self.length-1)      
                item=np.load(os.path.join(self.datapath,self.datalist[k]).strip())
                numf=item.shape[0]
                start=random.randint(0,numf-5)
                item=item[start:start+5,:,:,:]/255.0
                data.append(item)
            self.dataqueue.put(np.array(data))
    def getdata(self):
        return self.dataqueue.get()
    def close(self):
        self.on=False

class toframe(object):
    def __init__(self,path):
        self.path=path
        self.datalist=os.listdir(path)
     
    def run(self):
        for item in self.datalist:
            if mode=='dir':
                dirpath=os.path.join(self.path,item.strip('.avi'))
                if tf.gfile.Exists(dirpath):
                    break
                else:
                    tf.gfile.MkDir(dirpath) 
            elif mode=='npy':
                savef=[]
                avipath=os.path.join(self.path,item)
                cap=cv2.VideoCapture(avipath)
                print avipath
                ret=1
                while(ret):
                    ret, frame = cap.read()
                    if (ret):
                        try:
                            frame = cv2.resize(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),(64,64))
                            savef.append(frame)
                        except:
                            print 'noframe'
                savef=np.array(savef)
                np.save(avipath.replace('.avi','.npy'),savef)

def listinit(path):
    fop1=open('testicedancing.txt','w')
    fop2=open('trainicedancing.txt','w')
    for item in os.listdir(path):
        if item.endswith('.npy'):
            k=random.randint(0,9)
            if k==9:
                fop1.write(item+'\n')
            else:
                fop2.write(item+'\n')
#test code
'''
import datetime
a=dataloader('/ssd/10.10.20.21/share/guojiaming/UCF-101/Surfing','testsurfing.txt','test',16)

a.start()
time.sleep(1)
begin=datetime.datetime.now()
for i in xrange(6):
    print a.getdata().shape
end=datetime.datetime.now()
k=end-begin
print k.total_seconds()
a.close()
'''
listinit('/ssd/10.10.20.21/share/guojiaming/UCF-101/IceDancing')

