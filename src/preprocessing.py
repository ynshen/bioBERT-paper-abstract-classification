class Abstract(object):

    def __init__(self, label=None, title=None, abstract=None):
        if label is not None:
            self.label = label
        if title is not None:
            self.title = title
        if abstract is not None:
            self.abstract = abstract


class DataConverter:

    def __init__(self):
        self.dataset = None
        pass

    def load_ann(self, file_dirc, inplace=True):

        label_map = {
            'F': 0,
            'T': 1
        }
        dataset = []
        with open(file_dirc) as handle:
            while True:
                label = next(handle)
                if label is None:
                    break
                label = label_map(label[1])
                title = next(handle)
                abstract = next(handle)
                dataset.append(Abstract(label=label, title=title, abstract=abstract))

        self.dataset = dataset

    def to_tsv_full(self, output_dirc):

        if output_dirc.find('.tsv') == -1:
            output_dirc += 'dataset.tsv'

        with open(output_dirc, 'w') as handle:
            for abst in self.dataset:
                output_dirc.write(abst.)