class Confirmator:

    @staticmethod
    def confirm():
        val = input('Type "confirm" for confirmation:   ')
        if val != 'confirm':
            print('please review the dataset')
            exit()