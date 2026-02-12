"""
STUDENT GRADE ANALYSIS SYSTEM
==============================
A practical NumPy project for analyzing student performance

Features:
- Generate student data
- Calculate statistics (mean, median, std, etc.)
- Grade distribution analysis
- Performance comparison
- Identify top/bottom performers
- Visualize results with text-based charts
"""

import numpy as np

# ============================================================================
# 1. DATA GENERATION
# ============================================================================
print("=" * 70)
print("STUDENT GRADE ANALYSIS SYSTEM")
print("=" * 70)

# Set random seed for reproducibility
np.random.seed(42)

# Generate student data
num_students = 30
num_subjects = 5

# Subject names
subjects = ['Math', 'Physics', 'Chemistry', 'English', 'History']

# Generate random grades (0-100) for each student in each subject
grades = np.random.randint(40, 100, size=(num_students, num_subjects))

# Student names (simplified as Student 1, Student 2, etc.)
student_names = [f"Student {i + 1}" for i in range(num_students)]

print(f"\nGenerated data for {num_students} students across {num_subjects} subjects")
print(f"Subjects: {', '.join(subjects)}")
print("\nFirst 5 students' grades:")
print(f"{'Student':<12}", end="")
for subject in subjects:
    print(f"{subject:>10}", end="")
print()
print("-" * 70)

for i in range(5):
    print(f"{student_names[i]:<12}", end="")
    for grade in grades[i]:
        print(f"{grade:>10}", end="")
    print()

# ============================================================================
# 2. BASIC STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("OVERALL STATISTICS")
print("=" * 70)

overall_mean = np.mean(grades)
overall_median = np.median(grades)
overall_std = np.std(grades)
overall_min = np.min(grades)
overall_max = np.max(grades)

print(f"\nOverall Performance:")
print(f"  Mean Score:      {overall_mean:.2f}")
print(f"  Median Score:    {overall_median:.2f}")
print(f"  Std Deviation:   {overall_std:.2f}")
print(f"  Minimum Score:   {overall_min}")
print(f"  Maximum Score:   {overall_max}")

# ============================================================================
# 3. SUBJECT-WISE ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("SUBJECT-WISE ANALYSIS")
print("=" * 70)

# Calculate statistics for each subject (along axis 0 - down the columns)
subject_means = np.mean(grades, axis=0)
subject_medians = np.median(grades, axis=0)
subject_std = np.std(grades, axis=0)
subject_min = np.min(grades, axis=0)
subject_max = np.max(grades, axis=0)

print(f"\n{'Subject':<12} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 70)
for i, subject in enumerate(subjects):
    print(f"{subject:<12} {subject_means[i]:>8.2f} {subject_medians[i]:>8.2f} "
          f"{subject_std[i]:>8.2f} {subject_min[i]:>8} {subject_max[i]:>8}")

# Find best and worst performing subjects
best_subject_idx = np.argmax(subject_means)
worst_subject_idx = np.argmin(subject_means)

print(f"\nBest performing subject:  {subjects[best_subject_idx]} (Avg: {subject_means[best_subject_idx]:.2f})")
print(f"Worst performing subject: {subjects[worst_subject_idx]} (Avg: {subject_means[worst_subject_idx]:.2f})")

# ============================================================================
# 4. STUDENT-WISE ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("STUDENT-WISE ANALYSIS")
print("=" * 70)

# Calculate statistics for each student (along axis 1 - across the rows)
student_averages = np.mean(grades, axis=1)
student_totals = np.sum(grades, axis=1)
student_min = np.min(grades, axis=1)
student_max = np.max(grades, axis=1)

# Find top 5 students
top_5_indices = np.argsort(student_averages)[-5:][::-1]  # Get indices of top 5
print("\nTop 5 Students:")
print(f"{'Rank':<6} {'Student':<12} {'Average':>10} {'Total':>10}")
print("-" * 40)
for rank, idx in enumerate(top_5_indices, 1):
    print(f"{rank:<6} {student_names[idx]:<12} {student_averages[idx]:>10.2f} {student_totals[idx]:>10}")

# Find bottom 5 students
bottom_5_indices = np.argsort(student_averages)[:5]
print("\nBottom 5 Students:")
print(f"{'Rank':<6} {'Student':<12} {'Average':>10} {'Total':>10}")
print("-" * 40)
for rank, idx in enumerate(bottom_5_indices, 1):
    print(f"{rank:<6} {student_names[idx]:<12} {student_averages[idx]:>10.2f} {student_totals[idx]:>10}")

# ============================================================================
# 5. GRADE DISTRIBUTION
# ============================================================================
print("\n" + "=" * 70)
print("GRADE DISTRIBUTION")
print("=" * 70)


# Define grade ranges
def assign_letter_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'


# Count grades in each category
grade_a = np.sum(grades >= 90)
grade_b = np.sum((grades >= 80) & (grades < 90))
grade_c = np.sum((grades >= 70) & (grades < 80))
grade_d = np.sum((grades >= 60) & (grades < 70))
grade_f = np.sum(grades < 60)

total_grades = grades.size

print(f"\nGrade Distribution (out of {total_grades} total grades):")
print(f"{'Grade':<10} {'Count':>10} {'Percentage':>12} {'Chart':<30}")
print("-" * 70)

grades_dict = {
    'A (90-100)': grade_a,
    'B (80-89)': grade_b,
    'C (70-79)': grade_c,
    'D (60-69)': grade_d,
    'F (0-59)': grade_f
}

for grade_range, count in grades_dict.items():
    percentage = (count / total_grades) * 100
    bar = '█' * int(percentage / 2)  # Scale down for display
    print(f"{grade_range:<10} {count:>10} {percentage:>11.1f}% {bar}")

# ============================================================================
# 6. PERFORMANCE CATEGORIES
# ============================================================================
print("\n" + "=" * 70)
print("STUDENT PERFORMANCE CATEGORIES")
print("=" * 70)

# Categorize students based on their average
excellent = student_averages >= 90
good = (student_averages >= 75) & (student_averages < 90)
average = (student_averages >= 60) & (student_averages < 75)
poor = student_averages < 60

print(f"\nPerformance Categories:")
print(f"  Excellent (≥90):     {np.sum(excellent)} students")
print(f"  Good (75-89):        {np.sum(good)} students")
print(f"  Average (60-74):     {np.sum(average)} students")
print(f"  Needs Improvement:   {np.sum(poor)} students")

# ============================================================================
# 7. SUBJECT CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("SUBJECT CORRELATION ANALYSIS")
print("=" * 70)

# Calculate correlation matrix
correlation_matrix = np.corrcoef(grades.T)

print("\nCorrelation between subjects:")
print(f"{'':>12}", end="")
for subject in subjects:
    print(f"{subject:>10}", end="")
print()
print("-" * 70)

for i, subject in enumerate(subjects):
    print(f"{subject:>12}", end="")
    for j in range(len(subjects)):
        print(f"{correlation_matrix[i, j]:>10.2f}", end="")
    print()

# Find most correlated subjects (excluding diagonal)
correlation_no_diag = correlation_matrix.copy()
np.fill_diagonal(correlation_no_diag, 0)
max_corr_idx = np.unravel_index(np.argmax(correlation_no_diag), correlation_no_diag.shape)

print(f"\nMost correlated subjects: {subjects[max_corr_idx[0]]} and {subjects[max_corr_idx[1]]} "
      f"(correlation: {correlation_matrix[max_corr_idx]:.2f})")

# ============================================================================
# 8. IMPROVEMENT ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("IMPROVEMENT RECOMMENDATIONS")
print("=" * 70)

# Find students who need help in specific subjects
print("\nStudents needing help by subject:")
for i, subject in enumerate(subjects):
    failing_students = grades[:, i] < 60
    num_failing = np.sum(failing_students)

    if num_failing > 0:
        print(f"\n{subject}: {num_failing} student(s) with score < 60")
        failing_indices = np.where(failing_students)[0]
        for idx in failing_indices[:3]:  # Show first 3
            print(f"  - {student_names[idx]}: {grades[idx, i]}")
    else:
        print(f"\n{subject}: All students passing (≥60)")

# ============================================================================
# 9. STATISTICAL INSIGHTS
# ============================================================================
print("\n" + "=" * 70)
print("STATISTICAL INSIGHTS")
print("=" * 70)

# Find students with high variance (inconsistent performance)
student_variance = np.var(grades, axis=1)
most_consistent_idx = np.argmin(student_variance)
most_inconsistent_idx = np.argmax(student_variance)

print(f"\nMost consistent student:   {student_names[most_consistent_idx]} "
      f"(variance: {student_variance[most_consistent_idx]:.2f})")
print(f"  Grades: {grades[most_consistent_idx]}")

print(f"\nMost inconsistent student: {student_names[most_inconsistent_idx]} "
      f"(variance: {student_variance[most_inconsistent_idx]:.2f})")
print(f"  Grades: {grades[most_inconsistent_idx]}")

# Percentile analysis
p25 = np.percentile(student_averages, 25)
p50 = np.percentile(student_averages, 50)
p75 = np.percentile(student_averages, 75)

print(f"\nPercentile Analysis of Student Averages:")
print(f"  25th Percentile: {p25:.2f}")
print(f"  50th Percentile: {p50:.2f}")
print(f"  75th Percentile: {p75:.2f}")

# ============================================================================
# 10. SAVE REPORT TO FILE
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING DETAILED REPORT")
print("=" * 70)

# Create a detailed report
report_file = "student_report.txt"

with open(report_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("STUDENT GRADE ANALYSIS REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write("COMPLETE STUDENT GRADES\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Student':<15}", )
    for subject in subjects:
        f.write(f"{subject:>10}")
    f.write(f"{'Average':>10}\n")
    f.write("-" * 70 + "\n")

    for i in range(num_students):
        f.write(f"{student_names[i]:<15}")
        for grade in grades[i]:
            f.write(f"{grade:>10}")
        f.write(f"{student_averages[i]:>10.2f}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("SUMMARY STATISTICS\n")
    f.write("=" * 70 + "\n")
    f.write(f"Total Students: {num_students}\n")
    f.write(f"Total Subjects: {num_subjects}\n")
    f.write(f"Overall Average: {overall_mean:.2f}\n")
    f.write(f"Overall Median: {overall_median:.2f}\n")
    f.write(f"Standard Deviation: {overall_std:.2f}\n")

print(f"\nDetailed report saved to: {report_file}")

# ============================================================================
# 11. EXPORT DATA AS CSV
# ============================================================================
csv_file = "student_grades.csv"

# Create header
header = "Student," + ",".join(subjects) + ",Average"

# Combine student names with grades and averages
data_with_names = np.column_stack((
    np.array(student_names).reshape(-1, 1),
    grades,
    student_averages.reshape(-1, 1)
))

# Save to CSV
np.savetxt(csv_file, data_with_names, delimiter=',', header=header,
           fmt='%s', comments='')

print(f"Data exported to CSV: {csv_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)

print("\nKey Findings:")
print(f"✓ Analyzed {num_students} students across {num_subjects} subjects")
print(f"✓ Overall class average: {overall_mean:.2f}")
print(f"✓ Best subject: {subjects[best_subject_idx]} ({subject_means[best_subject_idx]:.2f})")
print(f"✓ Top student: {student_names[top_5_indices[0]]} ({student_averages[top_5_indices[0]]:.2f})")
print(f"✓ {np.sum(excellent)} students performing excellently (≥90)")
print(f"✓ {np.sum(poor)} students need improvement (<60)")

print("\nFiles Generated:")
print(f"  - {report_file}")
print(f"  - {csv_file}")