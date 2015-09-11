from manifold.infrastructure import Retriever, Displayer

TITLE = '1. Retrieving and displaying data-sets'


def main():
    print(TITLE)

    # Instantiates the retriever referencing the file that contains
    # the data and marking that the data is delimited by commas.
    r = Retriever('../datasets/glass/glass.data', delimiter=',')

    # Last column gets separated from the rest
    # of the data and stored as target feature.
    r.split_target()

    # Retrieves data and the target.
    data, glass_type = r.retrieve()

    # Instantiates a Displayer, which will generate a image.
    Displayer() \
        .load('Glass data-set', data, glass_type) \
        .render()

    # A single displayer can load many data-sets and targets.
    # It will render multiple graphs in the same image.
    # Displayer().load(...).load(...).load(...)


if __name__ == '__main__':
    main()
