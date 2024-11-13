from IPython.display import HTML
from IPython.display import display
import numpy as np

#####

AA_COLOR = {'Y':'#ff9d00',
            'W':'#ff9d00',
            'F':'#ff9d00',
            'A':'#171616',
            'L':'#171616',
            'M':'#171616',
            'I':'#171616',
            'V':'#171616',
            'Q':'#04700d',
            'N':'#04700d',
            'S':'#04700d',
            'T':'#04700d',
            'H':'#04700d',
            'G':'#04700d',
            'E':'#ff0d0d',
            'D':'#ff0d0d',
            'R':'#2900f5',
            'K':'#2900f5',
            'C':'#ffe70d',
            'P':'#cf30b7'}

AA_CHARGE_COLOR = {'Y':'#171616',
            'W':'#171616',
            'F':'#171616',
            'A':'#171616',
            'L':'#171616',
            'M':'#171616',
            'I':'#171616',
            'V':'#171616',
            'Q':'#171616',
            'N':'#171616',
            'S':'#171616',
            'T':'#171616',
            'H':'#171616',
            'G':'#171616',
            'E':'#ff0d0d',
            'D':'#ff0d0d',
            'R':'#2900f5',
            'K':'#2900f5',
            'C':'#171616',
            'P':'#171616'}


#####


def styled_text(letters : str, colors : list[str], sizes : list):
    # Check if inputs are consistent
    if len(letters) != len(colors) or len(letters) != len(sizes):
        raise ValueError("The length of letters, colors, and sizes must be the same.")
    
    # Generate HTML string
    styled_html = "".join(
        f"<span style='color:{color}; font-size:{int(size)}px'>{letter}</span>"
        for letter, color, size in zip(letters, colors, sizes)
    )
    
    return HTML(styled_html)


def check_sequence_validity(seq: str):
    '''Checks that the sequence is a valid amino acid sequence'''
    #check that the sequence is all valid amino acids
    valid_letters = list(AA_COLOR.keys())
    for aa in seq:
        if aa not in valid_letters:
            raise Exception(f"Invalid amino acid {aa} was passed.")

def chemical_context_seq_plot(seq : str, scale_vec : np.ndarray,
                     min_font_sz = 8,
                     max_font_sz = 30):
    '''Produces a string of the amino acid sequence with colors for their chemical context with the size being based on the scale vector.
    the size is specified in em units.
    '''
    #check that the sequence is all valid amino acids
    check_sequence_validity(seq)

    #set the minimum font size to the minimum value of the scale vector
    #find the min and max values for the scale vector
    vec_min = np.min(scale_vec)
    vec_max = np.max(scale_vec)
    #find the slope of the matched line
    m = (max_font_sz - min_font_sz)/(vec_max-vec_min)

    #create a lambda function that matches the values
    matched_line = lambda x: m*(x-vec_min)+vec_min

    #match the values and turn them into a list
    sizes_vec = matched_line(scale_vec)
    #size_txt_list = [f"{sz}px" for sz in sizes_vec]

    #create the matched hex values for each amino acid
    color_list = [AA_COLOR[aa] for aa in seq]

    html_obj = styled_text(seq, color_list, sizes_vec.tolist())
    display(html_obj)


def color_aminoacids(seq : str, aminos_of_interest : str,
                    font_sz = 12):
    '''Produces the charge state of a sequence'''
    #check that the sequence is all valid amino acids
    check_sequence_validity(seq)

    #match the values and turn them into a list
    sizes_vec = [font_sz for k in range(len(seq))]
    #size_txt_list = [f"{sz}px" for sz in sizes_vec]

    #create the matched hex values for each amino acid
    color_list = [AA_COLOR[aa] if aa in aminos_of_interest else '#3f3f3f' for aa in seq]

    html_obj = styled_text(seq, color_list, sizes_vec)
    display(html_obj)