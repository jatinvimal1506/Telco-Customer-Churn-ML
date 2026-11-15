from typing import Tuple, List
import pandas as pd


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    Supports Great Expectations 1.x fluent API with graceful fallback.
    """
    print("🔍 Starting data validation with Great Expectations...")

    # Primary path: Great Expectations 1.x Fluent API
    try:
        import great_expectations as gx
        import great_expectations.expectations as gxe

        context = gx.get_context(mode="ephemeral")
        data_source = context.data_sources.add_pandas("pandas_source")
        data_asset = data_source.add_dataframe_asset(name="telco_asset")
        batch_def = data_asset.add_batch_definition_whole_dataframe("whole_df")
        batch = batch_def.get_batch(batch_parameters={"dataframe": df})

        suite = gx.ExpectationSuite(name="telco_suite")

        # Schema & Presence
        for col in [
            "customerID", "gender", "Partner", "Dependents",
            "PhoneService", "InternetService", "Contract",
            "tenure", "MonthlyCharges", "TotalCharges"
        ]:
            suite.add_expectation(gxe.ExpectColumnToExist(column=col))

        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="customerID"))

        # Value Sets
        suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="gender", value_set=["Male", "Female"]))
        suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="Partner", value_set=["Yes", "No"]))
        suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="Dependents", value_set=["Yes", "No"]))
        suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="PhoneService", value_set=["Yes", "No"]))
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(
                column="Contract",
                value_set=["Month-to-month", "One year", "Two year"]
            )
        )
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(
                column="InternetService",
                value_set=["DSL", "Fiber optic", "No"]
            )
        )

        # Numeric bounds
        suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120))
        suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0, max_value=200))
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="tenure"))
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"))

        results = batch.validate(suite)
        failed_expectations = [r.expectation_config.type for r in results.results if not r.success]
        
        total_checks = len(results.results)
        passed_checks = sum(1 for r in results.results if r.success)
        failed_checks = total_checks - passed_checks

        if results.success:
            print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
        else:
            print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
            print(f"   Failed expectations: {failed_expectations}")

        return results.success, failed_expectations

    except Exception as exc:
        print(f"⚠️ Great Expectations 1.x validation encountered error: {exc}. Trying fallback...")
        
        # Legacy Great Expectations fallback (< 1.0)
        try:
            import great_expectations as ge
            if hasattr(ge, "dataset") and hasattr(ge.dataset, "PandasDataset"):
                ge_df = ge.dataset.PandasDataset(df)
                ge_df.expect_column_to_exist("customerID")
                ge_df.expect_column_values_to_not_be_null("customerID")
                ge_df.expect_column_to_exist("gender")
                ge_df.expect_column_to_exist("Partner")
                ge_df.expect_column_to_exist("Dependents")
                ge_df.expect_column_to_exist("PhoneService")
                ge_df.expect_column_to_exist("InternetService")
                ge_df.expect_column_to_exist("Contract")
                ge_df.expect_column_to_exist("tenure")
                ge_df.expect_column_to_exist("MonthlyCharges")
                ge_df.expect_column_to_exist("TotalCharges")
                ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
                ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
                ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
                ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
                ge_df.expect_column_values_to_be_in_set("Contract", ["Month-to-month", "One year", "Two year"])
                ge_df.expect_column_values_to_be_in_set("InternetService", ["DSL", "Fiber optic", "No"])
                ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
                ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
                ge_df.expect_column_values_to_not_be_null("tenure")
                ge_df.expect_column_values_to_not_be_null("MonthlyCharges")
                
                res = ge_df.validate()
                failed = [r["expectation_config"]["expectation_type"] for r in res["results"] if not r["success"]]
                return res["success"], failed
        except Exception:
            pass

        # Standalone pandas-based check as final safety guard
        failed = []
        required = ["customerID", "gender", "Partner", "Dependents", 
                    "PhoneService", "InternetService", "Contract", 
                    "tenure", "MonthlyCharges", "TotalCharges"]
        for c in required:
            if c not in df.columns:
                failed.append(f"missing_{c}")
        if "customerID" in df.columns and df["customerID"].isnull().any():
            failed.append("null_customerID")
        if "gender" in df.columns and not df["gender"].isin(["Male", "Female"]).all():
            failed.append("invalid_gender")
        
        is_ok = len(failed) == 0
        if is_ok:
            print("✅ Data validation PASSED (fallback validation)")
        else:
            print(f"❌ Data validation FAILED (fallback validation): {failed}")
        return is_ok, failed

