""" exposes a function for accessing the experiment I keep in a public google sheet

The sheet is located at https://docs.google.com/spreadsheets/d/1tXdjVwmFV-xRws1_-lYPfye-MrtdhmwpVYyqNjgjrnQ.
This is where I keep Occnet experiment information, and it is used to populate the descriptions on the website to keep everything organized.
"""
# The ID and range of a sample spreadsheet.
# SAMPLE_SPREADSHEET_ID = '1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms'
# SAMPLE_RANGE_NAME = 'Class Data!A2:E'

from googleapiclient.discovery import build

SAMPLE_SPREADSHEET_ID = '1tXdjVwmFV-xRws1_-lYPfye-MrtdhmwpVYyqNjgjrnQ'
SAMPLE_RANGE_NAME = 'Experiments'

service = build('sheets', 'v4', credentials=None)

result = service.spreadsheets().values().get(
    spreadsheetId=SAMPLE_SPREADSHEET_ID, range=SAMPLE_RANGE_NAME).execute()
numRows = result.get('values') if result.get('values') is not None else 0
# print('{0} rows retrieved.'.format(numRows))

def get_experiment_descriptions():
    """ returns descriptions keyed by the experiment name
    """

    result = service.spreadsheets().values().get(
        spreadsheetId=SAMPLE_SPREADSHEET_ID, range=SAMPLE_RANGE_NAME).execute()
    numRows = result.get('values') if result.get('values') is not None else 0

    keys = numRows[1]

    experiment_descriptions = {}

    for i in range(2, len(numRows)):

        row = numRows[i]
        if row is None:
            break
        
        # assume the first key is "experiment"
        experiment = row[0]
        experiment_descriptions[experiment] = ""
        for j in range(1, len(keys)):
            if j >= len(row):
                break
            experiment_descriptions[experiment] += "[{}: {}] ".format(keys[j], row[j])

    return experiment_descriptions

# display the experiment data from the google sheet
# print(get_experiment_descriptions())