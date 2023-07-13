import slideflow as sf
from slideflow.mil import mil_config
import multiprocessing

def main():
    P = sf.Project('/home/people/PEOPLE2')
    P.annotations= '/home/people/PEOPLE2/annotations/all_annotations.csv'
    dataset= P.dataset(tile_px= 299, tile_um= '20x', sources=['PEOPLE2'])

    config = mil_config('transmil', lr=1e-3)
   
    weights= '/home/people/PEOPLE2/mil/test2-regularization/transpath/00005-transpath_train2_transmil_[0.001]'
    outcomes= 'CBR_4'
    bags= '/home/people/PEOPLE2/data/features/transpath_normed_torch_feats'
    act_path= weights+'/activations'
    out= 'csv'
    P.generate_mil_features(weights, config, dataset, outcomes, bags, act_path, out)


if __name__=='__main__':
    multiprocessing.freeze_support()
    main()