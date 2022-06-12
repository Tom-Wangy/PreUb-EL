import numpy as np

def binary(file):
    aminoacids='ARNDCQEGHILKMFPSTWYVX'
    aa2v={x:y for x,y in zip(aminoacids,np.eye(21,21).tolist())}
    #aa2v['-']=np.zeros(20)
    encoding=[]
    label=[]
    f=open(file,'r')
    for line in f:
        col=line.strip().split('\t')
        s=col[0]
        encoding.append([aa2v[x] for x in (s[0:20]+s[21:])])
        label.append(col[-1])
    f.close()
    encoding=np.array(encoding)
    label=np.array(label).astype('float32')
    return encoding,label

def aaindex(file):
    index=pd.read_table('aaindex31',sep='\s+',header=None)
    index=index.subtract(index.min(axis=1),axis=0).divide((index.max(axis=1)-index.min(axis=1)),axis=0)
    index=index.to_numpy().T
    index={x:y for x,y in zip('ARNDCQEGHILKMFPSTWYV',index.tolist())}
    index['X']=np.zeros(31).tolist()
    encoding=[]
    label=[]
    f=open(file,'r')
    for line in f:
        col=line.strip().split('\t')
        s=col[0]
        encoding.append([index[x] for x in (s[0:20]+s[21:])])
        label.append(col[-1])
    f.close()
    encoding=np.array(encoding)
    label=np.array(label).astype('float32')
    return encoding,label