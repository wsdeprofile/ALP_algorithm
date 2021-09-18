import argparse

def parse_args():
    main_parser = argparse.ArgumentParser()
    alg_parsers = main_parser.add_subparsers(title='algorithms of anchor link prediction', dest='alg')
    alg_parsers.add_parser('stul', help='STUL', parents=[parser_args_stul()])
    alg_parsers.add_parser('mna', help='MNA', parents=[parser_args_mna()])
    return main_parser.parse_args()

def parser_args_mna():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-ids", help="Path to ids",
                        default="data/ids.csv")
    parser.add_argument("-text_seq", help="Path to text_seq",
                        default="data/tt_data.json")
    parser.add_argument("-location_seq", help="Path to location_seq",
                        default="data/fq_data.json")
    parser.add_argument("-dataset_path", help="Path to user_dataset",
                        default="data/fq_tt_dataset.npy")
    parser.add_argument("-feature_path", help="Path to feature",
                        default="data/feature.csv")
    parser.add_argument("-train", help="train or not",
                        default=True)
    parser.add_argument("-save_model", help="save model or not",
                        default=True)
    parser.add_argument("-model_path", help="Path to model",
                        default="model/mna_model")
    return parser

def parser_args_stul():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-ids", help="Path to ids",
                        default="data/ids.csv")
    parser.add_argument("-twitter_seq", help="Path to twitter_seq",
                        default="data/twitter_seq.json")
    parser.add_argument("-flickr_seq", help="Path to flickr_seq",
                        default="data/flickr_seq.json")
    parser.add_argument("-save_feature", help="save_feature or not",
                        default=True)
    parser.add_argument("-feature_path", help="Path to feature",
                        default="data/feature.csv")
    parser.add_argument("-distThres", help="distThres",
                        default=0.2)
    parser.add_argument("-timeThres", help="distThres",
                        default=30 * 60000)
    parser.add_argument("-dc", help="dc",
                        default=0.5)
    parser.add_argument("-k", help="k",
                        default=2)
    parser.add_argument("-train", help="train or not",
                        default=True)
    parser.add_argument("-save_model", help="save model or not",
                        default=True)
    parser.add_argument("-model_path", help="Path to model",
                        default="model/best_model")
    parser.add_argument("-kernel", help="kernel of SVM",
                        default='rbf')
    parser.add_argument("-C", help="C of SVM",
                        default=1000)
    parser.add_argument("-gamma", help="gamma of SVM",
                        default=0.01)
    return parser

args = parse_args()