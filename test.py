import CustomImageGen

x_train_folders = CustomImageGen.list_of_folders("train",5)
x_test_folders = CustomImageGen.list_of_folders("test",5)
print(len(x_train_folders))
print(len(x_test_folders))
x_train_files = CustomImageGen.list_of_files("NUbots360","train",folders=x_train_folders)
x_test_files = CustomImageGen.list_of_files("NUbots360","test",folders=x_test_folders)
print(len(x_train_files))
print(len(x_test_files))