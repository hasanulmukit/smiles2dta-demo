from rdkit import Chem
import numpy as np
import threading

class Iterator(object):
    
    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)
        if n < batch_size:
            raise ValueError('Input data length is shorter than batch_size\nAdjust batch_size')

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
    
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)




class SmilesIterator(Iterator):

    def __init__(self, x, y, smiles_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dtype=np.float32
                 ):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        self.x = np.asarray(x)

        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.smiles_data_generator = smiles_data_generator
        self.dtype = dtype
        super(SmilesIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):

       
        
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        
        batch_x = np.zeros(tuple([current_batch_size] + [ self.smiles_data_generator.pad, self.smiles_data_generator._charlen]), dtype=self.dtype)
        for i, j in enumerate(index_array):
            smiles = self.x[j:j+1]
            x = self.smiles_data_generator.transform(smiles)
            batch_x[i] = x

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y


class SmilesGenerator(object):

    def __init__(self, charset = '@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True, isomericSmiles=True, enum=True, canonical=False):
        self._charset = None
        self.charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    @property
    def charset(self):
        return self._charset
        
    @charset.setter
    def charset(self, charset):
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = dict((c,i) for i,c in enumerate(charset))
        self._int_to_char = dict((i,c) for i,c in enumerate(charset))
        
    def fit(self, smiles, extra_chars=[], extra_pad = 5):

        charset = set("".join(list(smiles)))
        self.charset = "".join(charset.union(set(extra_chars)))
        self.pad = max([len(smile) for smile in smiles]) + extra_pad
        
    def randomize_smiles(self, smiles):

        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

    def transform(self, smiles):

        one_hot =  np.zeros((smiles.shape[0], self.pad, self._charlen),dtype=np.int8)
        
        if self.leftpad:
            for i,ss in enumerate(smiles):
                if self.enumerate: ss = self.randomize_smiles(ss)
                l = len(ss)
                diff = self.pad - l
                for j,c in enumerate(ss):
                    one_hot[i,j+diff,self._char_to_int[c]] = 1
            return one_hot
        else:
            for i,ss in enumerate(smiles):
                if self.enumerate: ss = self.randomize_smiles(ss)
                for j,c in enumerate(ss):
                    one_hot[i,j,self._char_to_int[c]] = 1
            return one_hot

      
    def reverse_transform(self, vect):
      
        smiles = []
        for v in vect:
            v=v[v.sum(axis=1)==1]
            smile = "".join(self._int_to_char[i] for i in v.argmax(axis=1))
            smiles.append(smile)
        return np.array(smiles)
     
if __name__ == "__main__":
    smiles = np.array([ "CCC(=O)O[C@@]1(CC[NH+](C[C@H]1CC=C)C)c2ccccc2",
                        "CCC[S@@](=O)c1ccc2c(c1)[nH]/c(=N/C(=O)OC)/[nH]2"]*10
                        )
    sm_en = SmilesGenerator(canonical=True, enum=False)
    sm_en.fit(smiles, extra_chars=["\\"])
    v = sm_en.transform(smiles)
    transformed = sm_en.reverse_transform(v)
    if len(set(transformed)) > 2: print("Too many different canonical SMILES generated")
    
    sm_en.canonical = False
    sm_en.enumerate = True
    v2 = sm_en.transform(smiles)
    transformed = sm_en.reverse_transform(v2)
    if len(set(transformed)) < 3: print("Too few enumerated SMILES generated")

    reconstructed = sm_en.reverse_transform(v[0:5])
    for i, smile in enumerate(reconstructed):
        if smile != smiles[i]:
            print("Error in reconstruction %s %s"%(smile, smiles[i]))
            break
    
    import pandas as pd
    df = pd.DataFrame(smiles)
    v = sm_en.transform(df[0])
    if v.shape != (20, 52, 18): print("Possible error in pandas use")
    
    sm_it = SmilesIterator(smiles, np.array([1,2]*10), sm_en, batch_size=10, shuffle=True)
    X, y = sm_it.next()
    if sum(y==1) - sum(y==2) > 1:
        print("Unbalanced generation of batches")
    if len(X) != 10: print("Error in batchsize generation")



        
