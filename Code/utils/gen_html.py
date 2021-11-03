import webbrowser
import numpy as np
import argparse
import glob, os

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', type=str, required=True)
parser.add_argument('--vis_tool', type=str, default='')
args = parser.parse_args()

def button():
    print("""
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    .collapsible {
    background-color: #777;
    color: white;
    cursor: pointer;
    padding: 18px;
    width: 100%;
    border: none;
    text-align: left;
    outline: none;
    font-size: 15px;
    }

    .active, .collapsible:hover {
    background-color: #555;
    }

    .content {
    padding: 0 18px;
    display: none;
    overflow: hidden;
    background-color: #f1f1f1;
    }
    </style>
    </head>
    """)

    # Collapsible
    print("""
    <script>
    var coll = document.getElementsByClassName("collapsible");
    var i;

    for (i = 0; i < coll.length; i++) {
    coll[i].addEventListener("click", function() {
        this.classList.toggle("active");
        var content = this.nextElementSibling;
        if (content.style.display === "block") {
        content.style.display = "none";
        } else {
        content.style.display = "block";
        }
    });
    }
    </script>
    """)
if __name__ == '__main__':
    '''
    Create html page for results visualization
    Input :
        1. Folder-session
    Output :
        1. Link to tennis/checkerboard visualization
    '''

    visualizer = '/Code/vis_tools/tennis_visualizer{}/'.format(args.vis_tool)
    chk_url = '{}/checkerboard.html'.format(visualizer)
    tennis_url = '{}/index.html'.format(visualizer)

    file = sorted(glob.glob('{}/**/*.json'.format(args.json_path), recursive=True))
    tag = [f.split('/')[-3] for f in file]
    run = [f.split('/')[-2] for f in file]
    session = {}
    for i, f in enumerate(file):
        #print("idx : ", i)
        #print("file : ", f)
        #print("tag : ", tag[i])
        #print("run : ", run[i])
        if tag[i] not in session.keys():
            session[tag[i]] = {}
        session[tag[i]][run[i]] = []
        #print(session)

    for i, f in enumerate(file):
        session[tag[i]][run[i]].append(file[i])

    for sess in session.keys():
        print("<button type=\"button\" class=\"collapsible\">{}</button>".format(sess))
        print("<div class=\"content\">")
        for sub_sess in session[sess].keys():
            print("<button type=\"button\" class=\"collapsible\">{}</button>".format(sub_sess))
            print("<div class=\"content\">")
            print("""
                <style>
                table, th, td {
                border:1px solid black;
                }
                </style>
            """)
            print("<table style=\"width:100%\">")
            print("""
                <tr>
                    <th>Run</th>
                    <th>Checkerboard</th>
                    <th>Tennis</th>
                </tr>""")

            for run in session[sess][sub_sess]:
                rn = run.split('/')[-1]
                run = run[5:]
                # Create table
                chk_link = "<a href={}?input={}>{}</a>".format(chk_url, run, rn)
                tennis_link = "<a href={}?input={}>{}</a>".format(tennis_url, run, rn)
                print("""
                    <tr>
                        <td>{}</td>
                        <td>{}</td>
                        <td>{}</td>
                    </tr>""".format(rn, chk_link, tennis_link))
            print("</table>")
            print("</div>")

        print("</div>")
        print("<br>")

    button()
