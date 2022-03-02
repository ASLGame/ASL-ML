import csv

def logging_csv(number, landmark_list):

    csv_path = 'model/all_data.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])
    return