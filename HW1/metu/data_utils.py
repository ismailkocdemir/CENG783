from scipy.signal import argrelmax, argrelmin, find_peaks_cwt
import numpy as np
import os
import h5py

def load_dataset(filename):
    """Load your 'PPG to blood pressure' dataset"""
    # TODO: Fill this function so that your version of the data is loaded from a file into vectors    
    arrays = []
    with h5py.File(filename, 'r') as f:
        dataset = f['Part_1']
        indexes = np.random.choice(dataset.shape[0], 200) # Random 200 collection. 
        for i in indexes:
            data = dataset[i]
            name = h5py.h5r.get_name(data[0], f.id)
            if f[name].shape[0] < 40000 and f[name].shape[0] > 15000: # 3000 / 60 = 50 peaks minimum.
                arrays.append(np.array(f[name]))

    in_list = []
    out_list = []
    window_size = 1000
    
    print len(arrays)
    print [a.shape for a in arrays]

    j = 0
    for array in arrays:
        i  = 0
        print 'Now:', j , array.shape

        while i < 12000: #< array.shape[0] - window_size + 1:
            if i%500 == 0:
                print i
            in_list.append(array[i:i+window_size, 0])
            window_abg = array[i:i+window_size, 1]

            max_indexes = find_peaks_cwt(window_abg, [1])
            min_indexes = find_peaks_cwt(-1*window_abg, [1])

            max_avg = np.average( window_abg[max_indexes] )
            min_avg = np.average( window_abg[min_indexes] )
            out_list.append( [max_avg, min_avg] )
            i+= np.random.randint(20, 80) # Peak distance is 60 on average. Avoids windows starting from similar values.  

        j+=1
    X = np.array( in_list )
    y = np.array( out_list )
    del in_list
    del out_list
    print X.shape, y.shape
    np.save('long_in.npy', X)
    np.save('long_out.npy', y)
    return X, y

if __name__ == '__main__':
    # TODO: You can fill in the following part to test your function(s)/dataset from the command line
    filename = os.getcwd() + '/dataset/Part_1.mat'
    X, y = load_dataset(filename)

