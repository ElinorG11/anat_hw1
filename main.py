num_of_rows = int(input("Please insert number of rows:\n"))
for row_idx in range(num_of_rows+1):
  end_of_line = False
  for j in range(row_idx):
    print(" ", end='')
  for i in range(2*(num_of_rows-row_idx)+1):
    print('*', end='')
    end_of_line = True
  if end_of_line:
    print('\n')