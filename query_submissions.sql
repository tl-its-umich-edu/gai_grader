SELECT r_assessment.value.user_id as submitter_id, 
r_assessment.value.assessor_id as grader_id,
s.key.id as submission_id,
r_assessment.value.assessment_type,
r_assessment.value.score,
r_assessment.value.rubric_id,
a.value.context_id as canvas_course_id,
a.key.id as assignment_id,
a.value.title as assignment_title,
s.value.workflow_state as submission_workflow_state,
a.value.workflow_state as assignment_workflow_state,
r_assessment.value.artifact_attempt,
r_assessment.value.data as data
from `udp-umich-prod.canvas.rubric_assessments` r_assessment,
(
SELECT key.id as association_id, value.title
--, value.association_id as association_id, value.association_type
FROM `udp-umich-prod.canvas.rubric_associations` 
where value.context_type = 'Course'
--and value.title like '%ASSIGNMENT TITLE%'
and value.context_id =656488
and value.workflow_state='active'
order by title) r_association,
`udp-umich-prod.canvas.submissions` s
,`udp-umich-prod.canvas.assignments` a
where r_assessment.value.rubric_association_id = 
r_association.association_id
and r_assessment.value.artifact_type='Submission'
and r_assessment.value.artifact_id = s.key.id
and s.value.assignment_id = a.key.id
and a.value.workflow_state='published'
-- and a.key.id=<ASSIGNMENT_ID>
--and r_assessment.value.user_id not in (OPT_OUT_STUDENT1_CANVAS_ID, OPT_OUT_STUDENT2_CANVAS_ID)
order by r_assessment.value.rubric_id, r_assessment.value.user_id