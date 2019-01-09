import pickle
import numpy as np



class prediction():
    def __init__(self,tdata,wlist,ind_list):
        self.t = tdata
        self.lenkr = self.t.shape[1]
        self.w = wlist
        self.index = ind_list

    def predict(self,p):
        out = np.zeros((1,self.lenkr))

        for i in range(2): #flag
            for j in range(8): #makuragi
                index = self.index[i][j]
                w = self.w[i][j]
                if index != []:
                    for k in range(index.shape[0]):
                        ind = int(index[k])
                        w_ij = w[k]
                        x = self.t[-p:,ind]
                        out[0,ind] = w_ij[0] + np.sum(w_ij[1:]*x,axis=0)

        self.t = np.append(self.t,out,axis=0)


class data():
    def __init__(self):
        find = ["A","B","C","D"]
        self.fNum = len(find)
        f_wlist = "ar_w_list.binaryfile"
        f_ind_list = "index_list.binaryfile"
        self.tData = []
        self.wlist = self.file_load(f_wlist)
        self.ind_list = self.file_load(f_ind_list)

        for i in range(len(find)):
            fname = "track_tTrain_{}.binaryfile".format(fileind[no])
            self.tData.append(self.file_load(fname))


    def file_load(self,filename):
        f = open(filename,"rb")
        d = pickle.load(f)
        f.close()
        return d


if __name__ == "__main__":
    myData = data()
    pre_day = 91
    p = 10

    output = []

    for i in renge(myData.fNum):
        pre = prediction(myData.tData,myData.wlist[i],myData.ind_list[i])
        for _ in range(pre_day):
            pre.predict(p)
        out = pre.tData[-pre_day:].reshape(1,pre_day*pre.lenkr)
        output.append(out)

    result = pd.DataFrame(output[0])
    for i in range(1,myData.fNum):
        result = pd.concat([result,output[i]],axis=0)
    result.index = result.shape[0]

    f = open("output_ar_split.csv","w")
    result.to_csv(f)
    f.close()
