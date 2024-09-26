import csv
from collections import defaultdict

collapsed_rows = defaultdict(list)

with open('Training_set.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        identifier_code = row['IDENTIFIER_CODE']
        collapsed_rows[identifier_code].append(row)

with open('collapsed.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
    writer.writeheader()
    
    for identifier_code, rows in collapsed_rows.items():
        row_dict = {key: val for row in rows for key, val in row.items()}
        writer.writerow(row_dict)