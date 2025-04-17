import src.data_utils as dutils
import pandas as pd


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
