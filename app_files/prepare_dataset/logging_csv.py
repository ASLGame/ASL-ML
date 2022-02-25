import csv

def logging_csv(number, landmark_list):

    csv_path = 'model/keypoint1.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])
    return