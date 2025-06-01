import pandas as pd
from lxml import etree


def xml_to_dataframe(input_files, tag='Record'):
    all_rows = []
    headers = None

    # Iterate over all input XML files
    for input_file in input_files:
        # Open the input XML file and parse it incrementally
        context = etree.iterparse(input_file, events=("end",), tag=tag, recover=True, huge_tree=True)

        # Iterate through the XML elements in the current file
        for _, elem in context:
            if headers is None:
                headers = list(elem.attrib.keys())  # Get the column headers
            row = [elem.attrib.get(h, "") for h in headers]
            all_rows.append(row)

            # Clear the element to save memory
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

    # Create a DataFrame from all rows and headers
    df = pd.DataFrame(all_rows, columns=headers)
    print(f"XML data has been merged into a DataFrame.")

    return df