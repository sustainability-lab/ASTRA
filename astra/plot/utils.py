import matplotlib.pyplot as plt
import seaborn as sns

def latexify(width, height, font_size, latex=True):
    """
    width: float
        Width of the figure in inches
        - For most single column paper formats, widths around 5 to 6.5 inches work fine.
        - For most double column paper formats, widths around 2.5 to 3.5 inches work fine.
        
    height: float
        Height of the figure in inches
        - Mostly 1.5 to 2 inch height works fine.
        
    font_size: float
        Font size in points.
        - You may start experimenting with 10 points
        
    latex: bool
        If True, use latex to render the text.
        - If LaTeX is not installed, set this to False.
    """
    
    # use TrueType fonts so they are embedded
    # https://stackoverflow.com/questions/9054884/how-to-embed-fonts-in-pdfs-produced-by-matplotlib
    # https://jdhao.github.io/2018/01/18/mpl-plotting-notes-201801/
    plt.rcParams["pdf.fonttype"] = 42
    
    # These are set but you can override them after calling the function
    plt.rc("font", size=font_size)  # controls default text sizes
    plt.rc("axes", titlesize=font_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=font_size)  # legend fontsize
    plt.rc("figure", titlesize=font_size)  # fontsize of the figure title
    
    # latexify: https://nipunbatra.github.io/blog/visualisation/2014/06/02/latexify.html
    plt.rcParams["backend"] = "ps"
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("figure", figsize=(width, height))