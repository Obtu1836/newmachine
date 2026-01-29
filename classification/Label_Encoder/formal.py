import numpy as np 


class Encoder:
    
    def fit(self,tags):
        tags=np.array(tags,dtype=int)
        self.uni_value=np.unique(tags)
        idx=np.arange(len(self.uni_value))

        self.mask=np.zeros(max(tags)+1,dtype=int)
        self.mask[self.uni_value]=idx

        return self

    def transform(self,labels):

        return self.mask[labels]
    
    def inverse_transform(self,tag):

        return self.uni_value[tag]



tags=[3,5,8,5,8,3,2,9,10,1]
encoder=Encoder()
encoder.fit(tags)
yp=encoder.transform(tags)

print((encoder.inverse_transform(yp)==tags).all())

