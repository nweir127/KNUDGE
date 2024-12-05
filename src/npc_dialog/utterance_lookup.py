import glob
import pickle

import pandas as pd


class UtteranceIndex:
    def __init__(self):
        self.index = pickle.load(open('data/OuterWorlds/utterance_lookup.pkl', 'rb'))
        self.index_df = pd.DataFrame(self.index.values())
        self.index_df['utterance_lower'] = self.index_df.utterance.apply(lambda x: x.lower())

    def __getitem__(self, item):
        return self.index[item]

    def text_lookup(self, in_text):
        in_text = in_text.lower()
        return self.index_df[self.index_df.utterance_lower.apply(lambda x: in_text in x)]



utterance_index = UtteranceIndex()

if __name__ == '__main__':
    utterance_dict = {}

    for file in glob.glob("data/OuterWorlds/TheOuterWorlds_Text/Text/asian_csvs/*.csv"):
        df = pd.read_csv(file).dropna(subset=['English'])
        df['conv'] = df['Table'].apply(lambda x: x.split(" ")[0].split("\\")[-1])
        for i, row in df.iterrows():
            assert (row.conv, row.ID) not in utterance_dict
            utterance_dict[row.conv, row.ID] = dict(utterance=row['English'],
                                                    conversation=row.conv,
                                                    ID=row.ID,
                                                    speaker=row['Speaker'],
                                                    listener=row['Listener'],
                                                    line_type=row['Line Type'])


    pickle.dump(utterance_dict, open("data/OuterWorlds/utterance_lookup.pkl", 'wb'))

