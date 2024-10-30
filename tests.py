import sys
import glob
import imageio


def test(path,name):
    
    ims = sorted(glob.glob(path+"/*rgb.png"))
    imgs = []
    for im in ims[:120]:
        imgs.append(imageio.imread(im))
        
    imageio.mimsave(name+'.gif', imgs, duration=1000*1/60,loop=0)




if __name__ == '__main__':
    test(sys.argv[1],sys.argv[2])
