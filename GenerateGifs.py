import imageio
import glob

def generate_gif(folder_in, path_out):
    name = folder_in.split('/')[-1]
    img_path = folder_in + '/*.png'
    gif_path = '{}/{}.gif'.format(path_out, name)

    with imageio.get_writer(gif_path, mode="I") as writer:
        fileNames = sorted(glob.glob(img_path))
        print("Generating Gif({}):".format(len(fileNames)), name, img_path, gif_path)
        for filename in fileNames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


if __name__ == '__main__':
    names = {"07U1fSrk9oI_1","0Rn5D2-kMTY_1","0Rn5D2-kMTY_2","0_eqVxGOjuI_1"}
    inpath = "checkpoints/DTV_Sky/220316133416/results/30_f/"
    outpath = "testout"
    final = [inpath+name for name in names]
    print(final)
    for name in names:
        folder_in = inpath+name
        file_out = "testout"
        generate_gif(folder_in, file_out)