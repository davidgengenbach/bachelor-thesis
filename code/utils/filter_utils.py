def file_should_be_processed(file, include_filter, exclude_filter, dataset,limit_dataset):
    """Returns true, if file is included AND not excluded AND in the limited datasets.
    
    Args:
        file (str): the file to be processed
        include_filter (str): string that has to be in `file` (can be None)
        exclude_filter (str): string that must not be in `file` (can be None)
        dataset (str): the dataset of the file
        limit_dataset (list(str)): the datasets that have been limited (= are allowed)
    
    Returns:
        bool: Whether the file should be processed
    """
    is_in_limited_datasets = (not limit_dataset or dataset in limit_dataset)
    is_included = (not include_filter or include_filter in file)
    is_excluded = (exclude_filter and exclude_filter in file)
    return is_in_limited_datasets and is_included and not is_excluded