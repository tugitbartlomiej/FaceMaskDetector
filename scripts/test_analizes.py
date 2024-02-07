import pandas as pd
import matplotlib.pyplot as plt

detection_log_df = pd.read_csv('/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/'
                               'metamorphic_test_detection/yolov8/detection_log.csv')

if not detection_log_df.empty:
    grouped = detection_log_df.groupby(['Folder', 'Detected']).size().unstack(fill_value=0)
    bar_width = 0.35
    r1 = grouped.index
    r2 = [x + bar_width for x in range(len(r1))]

    plt.figure(figsize=(15, 10))
    plt.bar(r1, grouped['No'], width=bar_width, label='Not Detected', color='r')
    plt.bar(r2, grouped['Yes'], width=bar_width, label='Detected', color='g')

    plt.xlabel('Folder')
    plt.ylabel('Number of Images')
    plt.title('Detection Results per Folder')
    plt.xticks([r + bar_width / 2 for r in range(len(r1))], r1, rotation=90)
    plt.legend()

    plt.tight_layout()
    plt.savefig('/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/detection_results_comparison.png')
    plt.show()
else:
    print("No data available for plotting.")
