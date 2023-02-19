# GEM_HOME:= /Library/Ruby/Gems/2.6.0
GEM_GEMS_DIRPATH?= $(GEM_HOME)/gems/
GEM_PLUGINS_DIRPATH?= $(GEM_HOME)/plugins/


GEM_THEME_DIRPATH?= $(GEM_GEMS_DIRPATH)$(GEM_THEME_NAMEVERSION)/
# GEM_THEME_NAMEVERSION?= minimal-mistakes-jekyll=4.24.0
GEM_THEME_NAMEVERSION?= minima-2.5.1

GEM_THEME_GREP_REGEX?= Quick

GEM_THEME_INCLUDE_FILEPATH?= $(GEM_THEME_INCLUDES_DIRPATH)$(GEM_THEME_INCLUDE_NAME).html
GEM_THEME_INCLUDE_NAME?= default
GEM_THEME_INCLUDES_DIRPATH?= $(GEM_THEME_DIRPATH)_includes/

GEM_THEME_LAYOUT_FILEPATH?= $(GEM_THEME_LAYOUTS_DIRPATH)$(GEM_THEME_LAYOUT_NAME).html
GEM_THEME_LAYOUT_NAME?= page
# GEM_THEME_LAYOUT_NAME?= default
# GEM_THEME_LAYOUT_NAME?= archive
# GEM_THEME_LAYOUT_NAME?= home
# GEM_THEME_LAYOUT_NAME?= single
GEM_THEME_LAYOUTS_DIRPATH?= $(GEM_THEME_DIRPATH)_layouts/

GEM_THEME_DATA_DIRPATH?= $(GEM_THEME_DIRPATH)_data/
GEM_THEME_SASS_DIRPATH?= $(GEM_THEME_DIRPATH)_sass/

serve:
	bundle exec jekyll serve

lock_build_environment:
	# For error in GitHub workflow
	bundle lock --add-platform x86_64-linux

show_build_environment:
	bundle env

edit_navigation:
	$(EDITOR) ./_data/navigation.yml

grep_theme:
	grep -r $(GEM_THEME_GREP_REGEX) $(GEM_THEME_DIRPATH)*

install_gems:
	# Install new packages
	bundle install

show_theme:
	tree  $(GEM_THEME_DIRPATH)

list_gems:
	ls -al $(GEM_GEMS_DIRPATH)

list_home:
	ls -al $(GEM_HOME)

list_theme_assets:
	tree  $(GEM_THEME_DIRPATH)assets/

list_theme_data:
	tree  $(GEM_THEME_DATA_DIRPATH)

list_theme_includes:
	tree  $(GEM_THEME_INCLUDES_DIRPATH)

list_theme_sass:
	tree  $(GEM_THEME_SASS_DIRPATH)

show_theme_include:
	cat  $(GEM_THEME_INCLUDE_FILEPATH)

show_theme_layout:
	cat  $(GEM_THEME_LAYOUT_FILEPATH)

list_theme_layouts:
	tree  $(GEM_THEME_LAYOUTS_DIRPATH)

update:
	# Install new packages and update existing ones
	bundle update
