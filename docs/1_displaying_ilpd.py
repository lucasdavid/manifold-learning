from manifold.infrastructure import Retriever, Displayer

from docs.base import Example


class DisplayingILPDExample(Example):
    title = '1. Retrieving and displaying the ILPD data set'

    def run(self):
        # Instantiates the retriever referencing the file that contains
        # the data and marking that the data is delimited by commas.
        r = Retriever('../datasets/liver/Indian Liver Patient Dataset (ILPD).csv', delimiter=',')

        # The last column gets separated from the rest of the data and stored as target feature.
        # Finally, retrieves feature vectors and the glass_type (i.e. target feature).
        data, glass_type = r.split_target().retrieve()

        # Instantiates a Displayer, which will generate a image.
        d = Displayer(t=self.title)

        for i in range(0, data.shape[1], 3):
            d.load(data[:, i:i + 3], glass_type, title='Glass data-set - features [%i, %i]' % (i, i + 3))

        d.render()


if __name__ == '__main__':
    DisplayingILPDExample().start()