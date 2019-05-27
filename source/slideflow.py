import argparse
import os

class SlideFlowProject:

	PROJECT_DIR = ""

	def __init__(self, project_folder):
		print('''SlideFlow v1.0\n==============\n''')
		print('''Loading project...''')
		if project_folder and not os.path.exists(project_folder):
			if self.yes_no_input(f'Directory "{project_folder}" does not exist. Create directory and set as project root? [Y/n] ', default=True):
				os.mkdir(project_folder)
			else:
				project_folder = self.dir_input("Where is the project root directory? ", create_on_invalid=True)
		if not project_folder:
			project_folder = self.dir_input("Where is the project root directory? ", create_on_invalid=True)
		self.PROJECT_DIR = project_folder

		project_json = os.path.join(project_folder, "project.json")
		if os.path.exists(project_json):
			self.load_project(project_json)
		else:
			self.create_project(project_json)

	def create_global_path(self, path_string):
		if path_string and (len(path_string) > 2) and path_string[:2] == "./":
			return os.path.join(self.PROJECT_DIR, path_string[2:])
		elif path_string and (path_string[0] != "/"):
			return os.path.join(self.PROJECT_DIR, path_string)
		else:
			return path_string

	def yes_no_input(self, prompt, default=None):
		yes = ['yes','y']
		no = ['no', 'n']
		while True:
			response = input(f"{prompt}")
			if not response and default:
				return default
			if response.lower() in yes:
				return True
			if response.lower() in no:
				return False
			print(f"Invalid response.")

	def dir_input(self, prompt, default=None, create_on_invalid=False):
		while True:
			response = self.create_global_path(input(f"{prompt}"))
			if not response and default:
				response = self.create_global_path(default)
			if not os.path.exists(response) and create_on_invalid:
				if self.yes_no_input(f'Directory "{response}" does not exist. Create directory? [Y/n] ', default=True):
					os.mkdir(response)
					return response
				else:
					continue
			elif not os.path.exists(response):
				print(f'Unable to locate directory "{response}"')
				continue
			return response

	def file_input(self, prompt, default=None, filetype=None):
		while True:
			response = self.create_global_path(input(f"{prompt}"))
			if not response and default:
				response = self.create_global_path(default)
			if not os.path.exists(response):
				print(f'Unable to locate file "{response}"')
				continue
			extension = response.split('.')[-1]
			if filetype and (extension != filetype):
				print(f'Incorrect filetype; provided file of type "{extension}", need type "{filetype}"')
				continue
			return response

	def int_input(self, prompt, default=None):
		while True:
			response = input(f"{prompt}")
			if not response and default:
				return default
			try:
				int_response = int(response)
			except ValueError:
				print("Please supply a valid number.")
				continue
			return int_response

	def load_project(self, project_json):
		if os.path.exists(project_json):
			print("Loaded project successfully")
		else:
			print("Unable to locate project json")

	def create_project(self, project_json):
		name = input("What is the project name? ")
		annotations = self.file_input("Where is the project annotations (CSV) file located? [./annotations.csv] ", 
									default='./annotations.csv', filetype="csv")
		slides = self.dir_input("Where are the SVS slides stored? [./slides] ",
									default='./slides', create_on_invalid=True)
		tiles = self.dir_input("Where are the tessellated image tiles stored? [./tiles] ",
									default='./tiles', create_on_invalid=True)
		models = self.dir_input("Where are the saved models stored? [./models] ",
									default='./models', create_on_invalid=True)
		tile_um = self.int_input("What is the tile width in microns? [280] ", default=280)
		tile_px = self.int_input("What is the tile width in pixels? [512] ", default=512)

		print("\nCreating project with the following parameters:\n")
		print(f"Project name: {name}")
		print(f"Project root: {self.PROJECT_DIR}")
		print(f"Project annotations file: {annotations}")
		print(f"Project slides location: {slides}")
		print(f"Project tiles location: {tiles}")
		print(f"Project models location: {models}")
		print(f"Project tile width (um): {tile_um}")
		print(f"Project tile width (px): {tile_px}")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Helper to guide through the SlideFlow pipeline")
	parser.add_argument('-p', '--project', help='Path to project directory.')
	args = parser.parse_args()

	project = SlideFlowProject(args.project)