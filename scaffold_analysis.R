library(vite)

input.files <- list.files(path='scaffold_analysis', full.names=TRUE)
col.names <- c('CD45', 'Ly6C', 'TCR', 'Ly6G', 'CD19', 'CD169', 'CD106', 'CD3', 'CD1632', 'CD8a', 'CD90', 'F480', 'CD11c', 'Ter119', 'CD11b', 'IgD', 'CD27', 'CD5', 'CD79b', 'CD71', 'CD31', 'CD4', 'IgM', 'B220', 'ERTR7', 'CD35', 'CD2135', 'CD44', 'NKp46')
input.files <- c("A.clustered.txt", "B.clustered.txt")
landmarks.data <- load_landmarks_from_dir("scaffold_input/", asinh.cofactor = 5, transform.data = T)
run_scaffold_analysis(input.files, ref.file = input.files[1], 
                        landmarks.data = landmarks.data, col.names = col.names)