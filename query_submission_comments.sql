SELECT s.value.user_id as submitter_id,
s.key.id as submission_id,
a.value.context_id as canvas_course_id,
a.key.id as assignment_id,
a.value.title as assignment_title,
sc.value.author_id as commenter_id,
sc.value.comment as submission_comments,
s.value.workflow_state as submission_workflow_state,
a.value.workflow_state as assignment_workflow_state
from
`udp-umich-prod.canvas.submissions` s,
`udp-umich-prod.canvas.submission_comments` sc,
`udp-umich-prod.canvas.assignments` a
where s.value.assignment_id = a.key.id
-- and a.value.title like '%ASSIGNMENT TITLE%'
and s.value.course_id = @course_id
and sc.value.submission_id = s.key.id
and a.value.workflow_state='published'
--and r_assessment.value.user_id not in (OPT_OUT_STUDENT1_CANVAS_ID, OPT_OUT_STUDENT2_CANVAS_ID)
order by s.value.user_id