import slideflow as sf
from slideflow.mil import mil_config
from slideflow.mil import eval_mil, generate_mil_features
import multiprocessing
import sys

def main():
    m= sys.argv[1]
    P = sf.Project('/home/people/PEOPLE2')
    P.annotations= '/home/people/PEOPLE2/annotations/all_annotations_test.csv'
    dataset= P.dataset(tile_px= 299, tile_um= '20x', sources=['PEOPLE2'])

    if m.lower()== 'transmil':
        config = mil_config('transmil', lr=1e-3)
        weights= '/home/people/PEOPLE2/mil/test2-regularization/transpath/00005-transpath_train2_transmil_[0.001]'
    elif m.lower()== 'attention_mil':
        config = mil_config('attention_mil', lr=1e-3)
        weights= '/home/people/PEOPLE2/mil/test2-regularization/transpath/00025-transpath-train2-attention0.0001'
    
    outcomes= 'CBR_4'
    bags= '/home/people/PEOPLE2/data/features/transpath_normed_torch_feats'

    #eval_mil(weights, dataset, outcomes, bags, config)

    activations= generate_mil_features(weights, config, dataset, outcomes, bags)
    df= activations.to_df()
    df.to_parquet(weights+"/out.parquet")
    df.to_csv(weights+"/out.csv")


if __name__=='__main__':
    multiprocessing.freeze_support()
    main()