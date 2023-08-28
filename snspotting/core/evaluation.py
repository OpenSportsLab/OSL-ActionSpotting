
from SoccerNet.Evaluation.ActionSpotting import evaluate


def evaluate_Spotting(cfg, GT_path, pred_path):

    # challenge sets to be tested on EvalAI
    if "challenge" in cfg.dataset.test.split: 
        print("Visit eval.ai to evaluate performances on Challenge set")
        return None
        
    results = evaluate(SoccerNet_path=GT_path, 
                Predictions_path=pred_path,
                split=cfg.dataset.test.split,
                prediction_file="results_spotting.json", 
                version=cfg.dataset.test.version)


    a_mAP = results["a_mAP"]
    a_mAP_per_class = results["a_mAP_per_class"]
    a_mAP_visible = results["a_mAP_visible"]
    a_mAP_per_class_visible = results["a_mAP_per_class_visible"]
    a_mAP_unshown = results["a_mAP_unshown"]
    a_mAP_per_class_unshown = results["a_mAP_per_class_unshown"]

    logging.info("Best Performance at end of training ")
    logging.info("a_mAP visibility all: " +  str(a_mAP))
    logging.info("a_mAP visibility all per class: " +  str( a_mAP_per_class))
    logging.info("a_mAP visibility visible: " +  str( a_mAP_visible))
    logging.info("a_mAP visibility visible per class: " +  str( a_mAP_per_class_visible))
    logging.info("a_mAP visibility unshown: " +  str( a_mAP_unshown))
    logging.info("a_mAP visibility unshown per class: " +  str( a_mAP_per_class_unshown))
    
    return results

