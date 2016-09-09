Contributing
================

Issues
--------
The GitHub Issues is used to track the development progress, issues,
bugs, feature requests.

However, the installation and usage related problems should NOT go to
the Issues, and the Wiki (TODO) or mailing lists (TODO) can be adopted
for those purposes.


Pull Requests
----------------
So you already have a patch, an improvement, or even a new feature?
It's GREAT!  And let's have it!

To make a nice pull request (PR), the following things should be noted:

* Please adhere to the [Code Guidelines](#code-guidelines) used
  throughout the project, and other requirements (e.g., test).
* The PR should generally go to the ``master`` branch;
  the ``gh-pages`` branch should be generated from the documentation
  source files.
* It is recommended to use *topic/feature branch* for a PR.
* A PR should focus on one clear thing, so making multiple smaller PRs
  is better than making one large PR.
* A PR should be made of many commits where appropriate, with each commit
  be a small, atomic change representing one step in the development.

The following procedure is the recommended way to get your work
incorporated in the project:

1. [Fork](https://help.github.com/fork-a-repo/) the project, clone your
   fork, and configure the remotes:

   ```sh
   # Clone your fork of the repo into the current directory
   git clone https://github.com/<your-username>/fg21sim.git
   # Navigate to the newly cloned directory
   cd fg21sim
   # Assign the original repo to a remote called "upstream"
   git remote add upstream https://github.com/liweitianux/fg21sim.git
   ```

2. If you cloned a while ago, get the latest changes from upstream:

   ```sh
   git checkout master
   git pull upstream master
   ```

3. Create a new topic branch (off the main project development branch)
   to contain your feature, change, or fix:

   ```sh
   git checkout -b <topic-branch-name>
   ```

4. Commit your changes in logical chunks. Please adhere to these
   [git commit message guidelines](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html).
   Use Git's [interactive rebase](https://help.github.com/articles/interactive-rebase)
   feature to tidy up your commits before making them public.

5. Locally merge (or rebase) the upstream development branch into your
   topic branch:

   ```sh
   git pull [--rebase] upstream master
   ```

6. Push your topic branch up to your fork:

   ```sh
   git push origin <topic-branch-name>
   ```

7. [Open a Pull Request](https://help.github.com/articles/using-pull-requests/)
    with a clear title and description against the `master` branch.


Code Guidelines
-------------------

### Python

Adhere to the [PEP 8](https://www.python.org/dev/peps/pep-0008) code style
and also take a look at this
[code style](http://docs.python-guide.org/en/latest/writing/style/) from
[The Hitchhiker's Guide to Python](http://docs.python-guide.org/).

It is strongly recommended to check the code with [``flake8``](https://gitlab.com/pycqa/flake8) which wraps ``pep8``, ``pyflakes``, and other checkers together.
So make sure it is installed before carrying on.

* Vim users:

  1. Recommend to install the [``spf13-vim``](https://github.com/spf13/spf13-vim)
     configuration distribution, which already integrates the great
     [``syntastic``](https://github.com/scrooloose/syntastic) syntax checking
     plugin;

  2. Add the following recommended settings to ``~/.vimrc.local``:

     ```vim
     let g:syntastic_always_populate_loc_list = 1
     let g:syntastic_auto_loc_list = 1
     let g:syntastic_check_on_open = 1
     let g:syntastic_check_on_wq = 0
     ```
   3. Check syntax manually with command ``:SyntasticCheck``, then browse
      the found errors/warnings in a subwindow with command ``:Errors``,
      and jump using commands ``:lnext`` and ``:lprev``.

* Emacs users:

  1. The [``spacemacs``](https://github.com/syl20bnr/spacemacs) is a great
     configuration distribution to use.

  2. Enable the [``syntax-checking``](https://github.com/syl20bnr/spacemacs/blob/master/layers/syntax-checking/README.org) layer,
     and you're ready to go.


License
-------
By contributing your code, you agree to license your contribution under
the [MIT License](LICENSE).

By contributing to the documentation, you agree to license your contribution
under the [Creative Commons Attribution 3.0 license](https://creativecommons.org/licenses/by/3.0/us/deed.en_US).
