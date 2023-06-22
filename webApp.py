import streamlit as st
from keras.models import load_model
import pandas as pd
import numpy as np


def predict(comptency, subjects, grades, loaded_model):
    data = pd.DataFrame({'Competency': comptency,
                         'Subjects': subjects,
                         'Grades': grades
                         })

    # Convert 'Subjects' column into separate columns
    subjects_df = pd.DataFrame(data['Subjects'].tolist())
    subjects_df.columns = [f'Subject_{i + 1}' for i in range(subjects_df.shape[1])]

    # Fill remaining subject columns with -1
    num_subjects = 9  # Number of subject columns
    remaining_subjects = num_subjects - subjects_df.shape[1]
    remaining_subject_columns = [f'Subject_{i + subjects_df.shape[1] + 1}' for i in range(remaining_subjects)]
    subjects_df = pd.concat([subjects_df, pd.DataFrame(np.nan, index=subjects_df.index, columns=remaining_subject_columns)],
                            axis=1)
    # Convert 'Grades' column into separate columns
    grades_df = pd.DataFrame(data['Grades'].tolist())
    grades_df.columns = [f'Grade_{i + 1}' for i in range(grades_df.shape[1])]

    # Fill remaining grade columns with NaN
    num_grades = 9  # Number of grade columns
    remaining_grades = num_grades - grades_df.shape[1]
    remaining_grade_columns = [f'Grade_{i + grades_df.shape[1] + 1}' for i in range(remaining_grades)]
    grades_df = pd.concat([grades_df, pd.DataFrame(np.nan, index=grades_df.index, columns=remaining_grade_columns)], axis=1)

    p_data = pd.concat([data[['Competency']], subjects_df, grades_df], axis=1)
    for col in ['Subject_1', 'Subject_2', 'Subject_3', 'Subject_4', 'Subject_5', 'Subject_6', 'Subject_7', 'Subject_8',
                'Subject_9', 'Competency']:
        p_data[col] = p_data[col].astype('category').cat.codes
    p_data.fillna(0, inplace=True)
    predictions = loaded_model.predict(p_data)

    y_pred_classes = np.argmax(predictions, axis=1)  # Obtain the predicted class labels
    return y_pred_classes


def main():
    st.title("ML Prediction App")
    st.markdown("""Это приложение позволяет прогнозировать уровень овладения компетенцией на основе выбранных предметов и оценок.

Приложение предлагает выбрать компетенцию из доступного списка и затем выбрать один или несколько предметов, связанных с выбранной компетенцией. Для каждого выбранного предмета необходимо указать оценку.""")

    # Load the model
    loaded_model = load_model('cnn_model.h5')

    # Define the competencies and their associated subjects
    competencies = {
        "P01": ['Иностранный язык', 'Казахский язык', 'Информационно-коммуникационные технологии',
                'Казахский (Русский) язык', 'Цифровые технологии по отраслям применения'],
        "P02": ['Информационно-коммуникационные технологии', 'Цифровые технологии по отраслям применения',
                         'Иностранный язык', 'Казахский (Русский) язык'],
        "P04": ['Физическая культура'],
        "P05": ['Физическая культура'],
        "P06": ['Дискретные структуры', 'Математика 1', 'Математика 2', 'Физика', 'Цифровая электроника',
                'Дискретная математика'],
        "P07": ['Алгоритмизация и программирование', 'Алгоритмы и структуры данных',
                'Архитектура и организация компьютерных систем', 'Компьютерные сети: проектирование и администрирование',
                'Производственная практика', 'Сетевые технологии', 'Системное программирование', 'IT инфраструктура',
                'Машинно-ориентированное программирование (язык Assembler)', 'Операционные системы'],
        "P08": ['Алгоритмизация и программирование', 'Алгоритмы и структуры данных', 'Производственная практика',
                'Учебная практика', 'Языки программирования 1: С, С++', 'Языки программирования 2: Java',
                'Языки программирования 3: R', 'Языки программирования 3: передовые языки', 'IT инфраструктура',
                'Языки программирования 1: GO',  'Языки программирования 2: динамические языки программирования',
                'Управление IT проектами', 'Инструментальные средства разработки программ',
                'Программирование на Java', 'Языки программирования 1: С#']
    }

    # Add UI components
    competency = st.selectbox("Выберите компетенцию", list(competencies.keys()))

    subjects = competencies[competency]
    selected_subjects = st.multiselect("Выберите предметы", subjects)

    selectedSubjects = []
    selectedGrade = []
    for subject in selected_subjects:
        grade = st.slider(f"Оценка по предмету {subject}", min_value=0, max_value=100, step=1)
        selectedSubjects.append(subject)
        selectedGrade.append(grade)

    if st.button("Прогноз"):
        with st.spinner("Выполняется прогноз..."):
            import time
            time.sleep(1.2)
            st.success("Прогноз завершен!")
        label_mapping = {
            0: "низкий",
            1: "ниже среднего",
            2: "средний",
            3: "высокий"
        }

        predictions = predict(competency, [selectedSubjects], [selectedGrade], loaded_model)
        predicted_label = label_mapping[predictions[0]]

        st.markdown(f"Прогнозируемый уровень овладения компетенцией: **{predicted_label}**")


if __name__ == '__main__':
    main()
