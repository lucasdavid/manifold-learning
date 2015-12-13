import numpy as np

from experiments.base import Experiment
from manifold.infrastructure import Retriever


class DisplayingILPDExperiment(Experiment):
    title = '3.1.2. Displaying ILPD Data Set'

    data_set_file = '../../datasets/liver/Indian Liver Patient Dataset (ILPD).csv'
    plotting = True

    def _run(self):
        # the data and marking that the data is delimited by commas.
        r = Retriever(self.data_set_file, delimiter=',')

        data, glass_type = r.split_target().retrieve()

        labels = [
            '\nAge',
            '\nGender',
            '\nTotal Bilirubin (TB)',
            '\nDirect Bilirubin (DB)',
            '\nAlkaline Phosphotase (Alk.)',
            '\nAlamine Aminotransferase (Sgpt)',
            '\nAspartate Aminotransferase (Sgot)',
            '\nTotal Protiens (TP)',
            '\nAlbumin (ALB)',
            '\nRatio Albumin and Globulin Ratio (A/G)'
        ]

        for i in range(0, data.shape[1], 3):
            self.displayer.load(data[:, i:i + 3], glass_type,
                                axis_labels=labels[i:i + 3])

        print('Correlation matrix:')
        print(np.corrcoef(data, rowvar=0))

        self.displayer.save('displaying_ilpd')


if __name__ == '__main__':
    DisplayingILPDExperiment().start()
