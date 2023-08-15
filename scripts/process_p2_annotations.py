"""Preprocesses annotated PartiPrompts decisions to keep only
those with high inter-annotator agreement.

Example usage:
python scripts/process_p2_annotations.py
"""

import collections


if __name__ == "__main__":
    # Load the annotated PartiPrompts.
    id2decision = {}

    with open('data/PartiPromptsAllDecisions.tsv', 'r') as f:
        outputs = f.readlines()
        
        for i in range(1, len(outputs)):
            votes = outputs[i].split('\t')[-1].strip().split(',')
            id2decision[i] = votes


    # # Filter Confident Examples
    # Set examples *without* high inter-annotator agreement to have the 'same' vote.
    id2vote = {}

    for p_id in id2decision:
        counts = collections.Counter(id2decision[p_id])
        if (counts['gen'] >= 4 or counts['ret'] >= 4) or \
           (counts['gen'] == 3 and counts['ret'] <= 1) or \
           (counts['ret'] == 3 and counts['gen'] <= 1):
            id2vote[p_id] = counts.most_common(1)[0][0]
        else:
            id2vote[p_id] = 'same'

    print(collections.Counter(id2vote.values()))

    output_path = 'data/PartiPromptsDecisionsConfident.tsv'
    with open(output_path, 'w') as wf:
        wf.write(outputs[0].replace('\tDecisions\n', '\tDecision\n'))
        for i in range(1, len(outputs)):
            # Remove last column.
            curr_data = outputs[i].split('\t')
            curr_data = '\t'.join(curr_data[:-1])
            # Add majority vote into the new column.
            curr_data += f'\t{id2vote[i]}'
            wf.write(curr_data + '\n')
    
    print('Saved to', output_path)