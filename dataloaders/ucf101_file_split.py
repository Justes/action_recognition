import os
import os.path

class UCF101FileSplit():

	def __init__(self):
		if os.path.exists("UCF101"):
			print("ucf101 data already splitted")
		else:
			self.UCF101()

	def get_train_test_lists(version='01'):
	    """
	    Using one of the train/test files (01, 02, or 03), get the filename
	    breakdowns we'll later use to move everything.
	    选择一个数据分割版本，并读取检索路径
	    """
	    # Get our files based on version. 
	    test_file = os.path.join('ucfTrainTestlist', 'testlist' + version + '.txt')
	    train_file = os.path.join('ucfTrainTestlist', 'trainlist' + version + '.txt')

	    # Build the test list.
	    with open(test_file) as fin:
	        test_list = [row.strip() for row in list(fin)]

	    # Build the train list. Extra step to remove the class index.
	    with open(train_file) as fin:
	        train_list = [row.strip() for row in list(fin)]
	        train_list = [row.split(' ')[0] for row in train_list]

	    # Set the groups in a dictionary.
	    file_groups = {
	        'train': train_list,
	        'test': test_list
	    }

	    return file_groups

	def move_files(file_groups):
	    """This assumes all of our files are currently in _this_ directory.
	    So move them to the appropriate spot. Only needs to happen once.
	    将视频文件移动到新建的路径中
	    """
	    # Do each of our groups.
	    for group, videos in file_groups.items():

	        # Do each of our videos.
	        for video in videos:

	            # Get the parts.
	            #parts = video.split(os.path.sep)
	            parts = video.split('/') 
	            classname = parts[0]
	            filename = parts[1]

	            # Check if this class exists.
	            if not os.path.exists(os.path.join(group, classname)):
	                print("Creating folder for %s/%s" % (group, classname))
	                os.makedirs(os.path.join(group, classname))

	            # Check if we have already moved this file, or at least that it
	            # exists to move
	            filename_input = os.path.join('UCF-101',classname, filename)
	            if not os.path.exists(filename_input):
	                print("Can't find %s to move. Skipping." % (filename))
	                continue

	            # Move it.
	            dest = os.path.join(group, classname, filename)
	            print("Moving %s to %s" % (filename, dest))
	            os.rename(filename_input, dest)

	    print("Done.")

	def UCF101():
	    """
	    Go through each of our train/test text files and move the videos
	    to the right place.
	    """
	    os.makedirs('test')
	    os.makedirs('train')
	    # Get the videos in groups so we can move them.
	    group_lists = get_train_test_lists()

	    # Move the files.
	    move_files(group_lists)

if __name__ == "__main__":
	FileSplit()