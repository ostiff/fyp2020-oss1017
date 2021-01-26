
# ---------------------------------------------------------
# TidyWidget
# ---------------------------------------------------------
# Pandas
import pandas as pd

# Libraries
from pandas.api.types import is_bool_dtype


# Helper methods.
def duplicated_combine_set(x):
    """This method combines rows.

    .. note: __repr__ of set orders values.

    The elements in x can be of any type, we cast
    them to the best possible type and we compute
    the mean if they are numeric, the max if they
    are boolean (hence keeping True) or the set of
    values otherwise (string).
    """
    try:
        # Convert dtypes
        x = x.convert_dtypes()
        # Boolean raise exception
        if is_bool_dtype(x.dtype):
            return x.max()
        # Return mean
        return pd.to_numeric(x).mean()
    except Exception as e:
        return ','.join(sorted(x))
        return set(x)


class TidyWidget:
    """This widget creates data in tidy structured.

    It receives data in the so-called stack structure and returns
    the data transformed in tidy structure.

    .. note: When combining duplicates it computes the mean
             for numeric dtypes and creates a set (csv) for other
             dtypes such as string or boolean.

    Examples
    --------
    # Create widget
    widget = TidyWidget(index=index, value=value)

    # Transform (keep all)
    transform, duplicated = \
        widget.transform(data, report_duplicated=True)

    # Transform (keep first)
    transform_first = \
        widget.transform(data, keep='first')

    Parameters
    ----------
    index: str or list, default ['id', 'date', 'column']
        The column names with the index. It will be used to
        identify duplicates within the data,.
    value: str, default, result
        The column name with the values
    convert_dtypes: boolean, default True
        Whether convert dtypes.
    reset_index: boolean, default True
        Whether reset index

    Returns
    -------
    """
    errors = {
        'True': True,
        'False': False
    }

    def __init__(self, index=['id', 'date', 'column'],
                 value='result', convert_dtypes=True,
                 reset_index=True, replace_errors=True):
        """Constructor"""
        # Add attributes
        self.index = index
        self.value = value
        self.convert_dtypes = convert_dtypes
        self.reset_index = reset_index

    def transform(self, data, report_duplicated=False, keep=False):
        """Transform stack data to tidy data.

        .. note: data = data.sort_values(by=['StudyNo', 'date', 'column'])

        Old code
        --------
        # Basic formatting
        #replace = {'result': {'False': False, 'True': True}}
        #tidy.date = pd.to_datetime(tidy.date)
        #tidy.date = tidy.date.dt.date
        #tidy = tidy.replace(replace)            # Quick fix str to bool
        #tidy = tidy.drop_duplicates()           # Drop duplicates
        #tidy = tidy.set_index(self.index)

        Parameters
        ----------
        data: pd.DataFrame
            The data in stacked format. It usually has the
            columns ['patient_id', 'date', 'column', 'result'].
            The first three are usually the index and the
            results used as values.

        report_duplicated: boolean, default False
            Whether to return a DataFrame with the duplicates.

        keep: str, default False
            Strategy to remove duplicates. The possible values are
            to keep 'first' appearance, to keep 'last' or to keep
            all appearances combining them in a set using 'False'

        Returns
        -------
        tidy: pd.DataFrame
            The tidy DataFrame

        report: pd.DataFrame
            The report with the duplicate rows.
        """
        # Copy data
        aux = data.copy(deep=True)

        # Remove columns that are not in index
        subset = self.index + [self.value]

        # Keep only interesting
        aux = aux[subset]
        aux = aux.drop_duplicates()
        aux = aux.set_index(self.index)

        # Replace errors
        aux.result = aux.result.replace(self.errors)

        # Look for index duplicates
        duplicated = \
            aux.index.duplicated(keep=keep)

        # Create duplicates
        combination = pd.DataFrame()

        if not keep:
            # Combine duplicates
            combination = aux[duplicated] \
                .groupby(self.index) \
                .result.apply(duplicated_combine_set) \
                .to_frame()

        # Create stack without duplicates
        tidy = pd.concat([aux[~duplicated], combination])
        tidy = tidy.sort_values(by=self.index)

        # Create tidy (pivot)
        tidy = tidy.unstack() \
            .droplevel(level=0, axis=1)

        if self.reset_index:
            tidy = tidy.reset_index()

        if self.convert_dtypes:
            tidy = tidy.convert_dtypes()

        # Return
        if report_duplicated:
            return tidy, aux[duplicated]
        return tidy