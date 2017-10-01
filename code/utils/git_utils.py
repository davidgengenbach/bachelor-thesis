#!/usr/bin/env python

import git

GIT_ROOT_DIR = '..'

def get_repo(directory = GIT_ROOT_DIR):
    repo = git.Repo('..')
    assert not repo.bare
    return repo

def get_current_commit(directory = GIT_ROOT_DIR):
    repo = get_repo(directory)
    return repo.commit()

if __name__ == '__main__':
    commit = get_current_commit()
    print(commit)