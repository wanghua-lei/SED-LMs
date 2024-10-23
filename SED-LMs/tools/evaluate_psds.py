import os
import pandas as pd
import re
import numpy as np
import warnings
import transformers
import sed_eval
warnings.filterwarnings("ignore")

def get_event_list_current_file(df, fname):
    """
    Get list of events for a given filename
    Args:
        df: pd.DataFrame, the dataframe to search on
        fname: the filename to extract the value from the dataframe
    Returns:
         list of events (dictionaries) for the given filename
    """
    event_file = df[df["filename"] == fname]
    if len(event_file) == 1:
        if pd.isna(event_file["event_label"].iloc[0]):
            event_list_for_current_file = [{"filename": fname}]
        else:
            event_list_for_current_file = event_file.to_dict("records")
    else:
        event_list_for_current_file = event_file.to_dict("records")

    return event_list_for_current_file


def event_based_evaluation_df(
    reference, estimated, t_collar=0.200, percentage_of_length=0.2
):
    """ Calculate EventBasedMetric given a reference and estimated dataframe

    Args:
        reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            reference events
        estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            estimated events to be compared with reference
        t_collar: float, in seconds, the number of time allowed on onsets and offsets
        percentage_of_length: float, between 0 and 1, the percentage of length of the file allowed on the offset
    Returns:
         sed_eval.sound_event.EventBasedMetrics with the scores
    """
    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling="zero_score",
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return event_based_metric

def segment_based_evaluation_df(reference, estimated, time_resolution=1.0):
    """ Calculate SegmentBasedMetrics given a reference and estimated dataframe

        Args:
            reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
                reference events
            estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
                estimated events to be compared with reference
            time_resolution: float, the time resolution of the segment based metric
        Returns:
             sed_eval.sound_event.SegmentBasedMetrics with the scores
        """
    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes, time_resolution=time_resolution
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        segment_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return segment_based_metric


def compute_sed_eval_metrics(predictions, groundtruth):
    """ Compute sed_eval metrics event based and segment based with default parameters used in the task.
    Args:
        predictions: pd.DataFrame, predictions dataframe
        groundtruth: pd.DataFrame, groundtruth dataframe
    Returns:
        tuple, (sed_eval.sound_event.EventBasedMetrics, sed_eval.sound_event.SegmentBasedMetrics)
    """
    metric_event = event_based_evaluation_df(
        groundtruth, predictions, t_collar=0.200, percentage_of_length=0.2
    )
    metric_segment = segment_based_evaluation_df(
        groundtruth, predictions, time_resolution=1.0
    )

    return metric_event, metric_segment


def decodetxt(file):
    #get the caption
    captions_pred = []
    captions_gt = []
    with open(file, 'r') as f:
        lines = f.readlines()
    file_names = []
    captions = []
    groundtruth =[]
    for line in lines:
        if line.startswith('Captions for file'):
            file_names.append(line.strip().replace('Captions for file: ', ''))
        if line.startswith('\t Predicted caption:'):
            captions.append(line.strip().replace('Predicted caption: ', ''))
        if line.startswith('\t groundths caption:'):
            groundtruth.append(line.strip().replace('groundths caption: ', ''))

    for pred_cap, gt_caps, f_name in zip(captions, groundtruth, file_names):
        captions_pred.append({'file_name': f_name, 'caption_predicted': pred_cap})
        ref_caps_dict = {'file_name': f_name}
        ref_caps_dict[f"caption"] = gt_caps
        captions_gt.append(ref_caps_dict)
    return captions_pred, captions_gt

# label = ['alarm', 'blender',"cat","dishes","dog","electric toothbrush","frying","running water","speech","vacuum cleaner"]
label = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

def generate_tomatch(captions_pred):
    captions_df = pd.DataFrame(captions_pred)
    predict_df = pd.DataFrame(columns=["filename", 'onset', 'offset', 'event_label'])

    for i in range(len(captions_df)):
        str = captions_df.iloc[i]["caption_predicted"]
        filename = captions_df.iloc[i]["file_name"]
        pattern = r'([\w\s]+)\s+heard\s+between\s+(\d+\.\d+)\s+and\s+(\d+\.\d+)\s+seconds'
        matches = re.findall(pattern, str)
        for match in matches:
            predict_df = predict_df.append({'filename':filename, 'event_label': match[0].strip().replace(' ','_',),
            'onset': float(match[1]), 'offset': float(match[2])}, ignore_index=True)
    # predict_df['event_label'].str.capitalize().replace("Alarm","Alarm_bell_ringing").replace("Electric_toothbrush","Electric_shaver_toothbrush")
    predict_df=predict_df[predict_df['event_label'].isin(label)].drop_duplicates()
    predict_df.to_csv('predict.tsv',sep='\t',index=False)
    return predict_df


def evaluate_psds(captions_pred, eval = False):
    if eval:
        gt = pd.read_csv('tools/PSDS_Eval/meta/eval.tsv',sep='\t')
        gt = pd.read_csv('tools/PSDS_Eval/meta/test.tsv',sep='\t')
    else:
        gt = pd.read_csv('tools/PSDS_Eval/meta/validation.tsv',sep='\t')
        gt = pd.read_csv('tools/PSDS_Eval/meta/val.tsv',sep='\t')

    preds_s = generate_tomatch(captions_pred)
    preds_s['filename'], preds_s['event_label'] = preds_s['event_label'],preds_s['filename']
    preds_s.rename(columns={'filename':'event_label','event_label':'filename'}, inplace=True)
    gt = gt.sort_values(by='filename')
    

    # calculate event-based and segment-based metrics
    save_dir="tools/PSDS_Eval/meta/metrics_test"

    event_metrics_s, segment_metrics_s = compute_sed_eval_metrics(preds_s, gt)

    with open(os.path.join(save_dir, "event_f1.txt"), "w") as f:
        f.write(str(event_metrics_s))

    with open(os.path.join(save_dir, "segment_f1.txt"), "w") as f:
        f.write(str(segment_metrics_s))

    macro_event= round(event_metrics_s.results()["class_wise_average"]["f_measure"]["f_measure"],3)
    macro_segment = round(segment_metrics_s.results()["class_wise_average"]["f_measure"]["f_measure"],3)
    return macro_event, macro_segment


if __name__ == '__main__':
    dir = "outputs/16k_urban/logging/captions_0ep_greedy.txt"
    predicted, ground = decodetxt(dir)
    # generate_tomatch(predicted).to_csv(dir+'1.tsv',sep='\t',index=False)
    # predicted_top,ground = decodetxt(dir+'captions_10ep_top9.txt')
    F1 = evaluate_psds(predicted,eval=True)
    print(F1)


