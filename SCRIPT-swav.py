import slideflow as sf
from slideflow import swav
import multiprocessing

def main():
  project_path = '/home/mattsacco/data/PROJECTS/UPenn_debug'
  project_path = '/home/mattsacco/PROJECTS/UPenn_debug'

  P = sf.Project(root=project_path)
  dataset = P.dataset(tile_px=299, tile_um=604)

  swav_args = swav.get_args(
    warmup_epochs=1,
    epochs=2,
    nmb_crops=[2, 6],
    size_crops=[299, 96],
    min_scale_crops=[0.14, 0.05],
    max_scale_crops=[1., 0.14],
    batch_size=32
  )

  P.train_swav(
    swav_args=swav_args,
    train_dataset=dataset,
  )

if __name__=='__main__':
    multiprocessing.freeze_support()
    main()
