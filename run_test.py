from test_seg.seg import seg
import glob

if __name__ == '__main__':
    paths = glob.glob(r"your_path")
    box = [113, 321, 65, 289, 168, 471]
    for path in paths:
        seg(path, r'your_path', box)
