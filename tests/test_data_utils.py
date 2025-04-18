from pathlib import Path
import src.data_utils as dutils
import pandas as pd


class TestLabelledTraitData:
    dataset = dutils.LabelledTraitData(
        data_path=Path("data"),
        var="N.Percent"
    )
    def test_train_data(self):
        """Test that the train data is loaded correctly."""
        # Check that the train data is not empty
        assert not self.dataset.train_data.empty

        # Check that the train labels are not empty
        assert not self.dataset.train_labels.empty
        # Check that the train data and labels have the same length
        assert len(self.dataset.train_data) == len(self.dataset.train_labels)
        # Check that the train labels are in the correct format
        assert self.dataset.train_labels.columns.tolist() == ['TraitValue']
        # Check that the train data and labels have the same index
        assert self.dataset.train_data.index.equals(self.dataset.train_labels.index)

        # Check data contains no NaN values
        assert not self.dataset.train_data.isna().any().any()
        # Check labels contains no NaN values
        assert not self.dataset.train_labels.isna().any().any()

    def test_val_data(self):
        """Test that the validation data is loaded correctly."""
        # Check that the validation data is not empty
        assert not self.dataset.val_data.empty

        # Check that the validation labels are not empty
        assert not self.dataset.val_labels.empty
        # Check that the validation data and labels have the same length
        assert len(self.dataset.val_data) == len(self.dataset.val_labels)
        # Check that the validation labels are in the correct format
        assert self.dataset.val_labels.columns.tolist() == ['TraitValue']
        # Check that the validation data and labels have the same index
        assert self.dataset.val_data.index.equals(self.dataset.val_labels.index)
        
        # Check data contains no NaN values
        assert not self.dataset.val_data.isna().any().any()
        # Check labels contains no NaN values
        assert not self.dataset.val_labels.isna().any().any()

    def test_test_data(self):
        """Test that the test data is loaded correctly."""
        # Check that the test data is not empty
        assert not self.dataset.test_data.empty

        # Check that the test labels are not empty
        assert not self.dataset.test_labels.empty
        # Check that the test data and labels have the same length
        assert len(self.dataset.test_data) == len(self.dataset.test_labels)
        # Check that the test labels are in the correct format
        assert self.dataset.test_labels.columns.tolist() == ['TraitValue']
        # Check that the test data and labels have the same index
        assert self.dataset.test_data.index.equals(self.dataset.test_labels.index)

        # Check data contains no NaN values
        assert not self.dataset.test_data.isna().any().any()
        # Check labels contains no NaN values
        assert not self.dataset.test_labels.isna().any().any()


class TestTrainValTestSplit:
    # Create a sample DataFrame
    df = pd.DataFrame({
        'A': range(10),
        'B': range(10, 20)
    })

    def test_not_empty_dataframes(self):
        """Test that the train, validation, and test sets are not empty."""
        # Split the DataFrame
        train_df, val_df, test_df = dutils.train_val_test_split(self.df)

        # Check that the DataFrames are not empty
        assert not train_df.empty
        assert not val_df.empty
        assert not test_df.empty

    def test_empty_intersection(self):
        """Test that the train, validation, and test sets do not intersect."""
        # Split the DataFrame
        train_df, val_df, test_df = dutils.train_val_test_split(self.df)

        # Check that the indices of the three sets do not intersect
        assert set(train_df.index) & set(val_df.index) & set(test_df.index) == set()

class TestGetOutlierZScore:
    def test_outlier_zscore(self):
        """Test that the outlier detection using Z-score works correctly."""
        # Create a sample Series
        data = pd.Series([1, 2, 3, 4, 5, 100])

        # Get outliers using Z-score
        # Std is heavily skewed by outlier, therefore we set zlim to 2 to detect it.
        outliers = dutils.get_outlier_zscore(data, zlim=2)

        # Check that the outlier is detected correctly
        assert len(outliers) == 1
        assert outliers.iloc[0] == 100

    def test_outlier_zscore_high_zlim(self):
        """Test that the outlier detection using Z-score works correctly."""
        # Create a sample Series
        data = pd.Series([1, 2, 3, 4, 5, 100])

        # Get outliers using Z-score
        # Std is heavily skewed by outlier, therefore zlim = 3 won't detect it.
        outliers = dutils.get_outlier_zscore(data, zlim=3)

        # Check that the outlier is not detected
        assert len(outliers) == 0

    def test_negative_mean(self):
        """Test that the outlier detection using Z-score works correctly with negative mean."""
        # Create a sample Series
        data = pd.Series([-1, -2, -3, -4, -5, -100])

        # Get outliers using Z-score
        # Std is heavily skewed by outlier, therefore we set zlim to 2 to detect it.
        outliers = dutils.get_outlier_zscore(data, zlim=2)

        # Check that the outlier is detected correctly
        assert len(outliers) == 1
        assert outliers.iloc[0] == -100

    def test_no_outliers(self):
        """Test that no outliers are detected when there are none."""
        # Create a sample Series
        data = pd.Series([1, 2, 3, 4, 5])

        # Get outliers using Z-score
        outliers = dutils.get_outlier_zscore(data)

        # Check that no outliers are detected
        assert len(outliers) == 0

class TestGetOutlierIQR:
    def test_outlier_iqr(self):
        """Test that the outlier detection using IQR works correctly."""
        # Create a sample Series
        data = pd.Series([1, 2, 3, 4, 5, 100])

        # Get outliers using IQR
        # IQR is not heavily skewed by outlier, therefore we can set zlim lower.
        outliers = dutils.get_outlier_iqr(data, zlim=1.5)

        # Check that the outlier is detected correctly
        assert len(outliers) == 1
        assert outliers.iloc[0] == 100

    def test_outlier_iqr_high_zlim(self):
        """Test that the outlier detection using IQR works correctly."""
        # Create a sample Series
        data = pd.Series([1, 2, 3, 4, 5, 100])

        # Check very high zlim = 50 doesn't return outliers.
        outliers = dutils.get_outlier_iqr(data, zlim=50)

        # Check that the outlier is not detected
        assert len(outliers) == 0

    def test_negative_mean(self):
        """Test that the outlier detection using IQR works correctly with negative mean."""
        # Create a sample Series
        data = pd.Series([-1, -2, -3, -4, -5, -100])

        # Get outliers using IQR
        # IQR is not heavily skewed by outlier, therefore we can set zlim lower.
        outliers = dutils.get_outlier_iqr(data, zlim=1.5)

        # Check that the outlier is detected correctly
        assert len(outliers) == 1
        assert outliers.iloc[0] == -100

    def test_no_outliers(self):
        """Test that no outliers are detected when there are none."""
        # Create a sample Series
        data = pd.Series([1, 2, 3, 4, 5])

        # Get outliers using IQR
        outliers = dutils.get_outlier_iqr(data)

        # Check that no outliers are detected
        assert len(outliers) == 0
