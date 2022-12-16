"""
Convert XVEC style ground truth to the default, e.g., for the SIFT dataset

XVEC: per row: (dim, vec)
DEFAULT: (dim, vec), rows

Usage: 
    python convert_gt.py --input /mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_groundtruth.ivecs --output /mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_groundtruth.ivecs.bin
"""
import numpy as np
import struct
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='/mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_groundtruth.ivecs', help="input file name")
parser.add_argument('--output', type=str, default='/mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_groundtruth.ivecs.bin', help="output file name")

args = parser.parse_args()

fname_in = args.input
fname_out = args.output

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    # Wenqi: Format of ground truth (for 10000 query vectors):
    #   1000(topK), [1000 ids]
    #   1000(topK), [1000 ids]
    #        ...     ...
    #   1000(topK), [1000 ids]
    # 10000 rows in total, 10000 * 1001 elements, 10000 * 1001 * 4 bytes
    return a.reshape(-1, d + 1)[:, 1:].copy()

def write_ibin(filename, vecs):
    """ Write an array of float32 vectors to *.fbin file
    Args:s
        :param filename (str): path to *.fbin file
        :param vecs (numpy.ndarray): array of float32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    nvecs, dim = vecs.shape
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('int32').flatten().tofile(f)

if __name__ == "__main__":
    vecs = ivecs_read(fname_in)
    write_ibin(fname_out, vecs)
